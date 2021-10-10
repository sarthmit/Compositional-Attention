import torch
import torch.nn
import torch.nn.functional as F
import math
from typing import Optional, Callable, List, Union, Tuple
from dataclasses import dataclass


@dataclass
class AttentionMask:
    src_length_mask: Optional[torch.Tensor]
    position_mask: Optional[torch.Tensor]


class CompositionalAttentionBase(torch.nn.Module):
    def __init__(self, state_size: int, n_heads: int, n_rules: int,  qk_dim: int, dot: bool = True, dropout: float = 0.1):
        assert state_size % n_heads == 0
        super().__init__()
        self.state_size = state_size
        self.projection_size = state_size // n_heads
        self.n_heads = n_heads
        self.n_rules = n_rules
        self.qk_dim = qk_dim
        self.dot = dot
        self.scale = 1.0 / math.sqrt(self.projection_size)
        self.query_value_net = torch.nn.Linear(state_size, self.qk_dim * self.n_heads)

        if self.dot:
            self.key_value_net = torch.nn.Linear(self.projection_size, self.qk_dim)
        else:
            self.score_network = torch.nn.Linear(self.projection_size + self.qk_dim, 1)

        self.dropout = torch.nn.Dropout(dropout)
        self.multi_head_merge = torch.nn.Linear(state_size, state_size, bias=False)

    def _masked_softmax(self, logits: torch.Tensor, mask: Optional[AttentionMask]) -> torch.Tensor:
        if mask is None or (mask.src_length_mask is None and mask.position_mask is None):
            return F.softmax(logits, -1)

        # Output shape: [n_batch * n_heads, n_time_dest, n_time_src]
        bb, n_heads, n_time_dest, n_time_src = logits.shape
        #
        # logits = logits.view(bb // self.n_heads, self.n_heads, n_time_dest, n_time_src)

        if mask.position_mask is not None:
            logits = logits.masked_fill(mask.position_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if mask.src_length_mask is not None:
            logits = logits.masked_fill(mask.src_length_mask.unsqueeze(1).unsqueeze(1), float("-inf"))

        logits = F.softmax(logits, -1)
        return logits.view(bb, n_heads, n_time_dest, n_time_src)

    def _attention_read(self, mask: Optional[AttentionMask], logits: torch.Tensor, v: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        # logits: [n_batch * n_heads, n_out, n_in]
        # v: [n_nbatch * n_heads, n_in]
        # Output data shape [n_batch * n_heads, n_time_dest, data_size]
        # Out attention score shape: [n_batch, n_heads, n_time_dest, n_time_src]

        scores = self._masked_softmax(logits, mask)
        scores = self.dropout(scores)
        return torch.bmm(scores, v), scores.view(-1, self.n_heads, *scores.shape[1:])

    def merged_attention(self, x,  mask, n_batch: int, n_read: int, q, k, v) -> \
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        score = self._masked_softmax(torch.matmul(q, k), mask).unsqueeze(2)  # (bsz, n_heads, n_read, n_write)

        out = torch.matmul(score, v)  # (bsz, n_heads, n_rules, n_read, att_dim)
        out = out.view(n_batch, self.n_heads, self.n_rules, n_read, self.projection_size)

        out = out.permute(0, 3, 1, 2, 4).reshape(n_batch, n_read, self.n_heads, self.n_rules, self.projection_size)

        if self.dot:
            q_v = self.query_value_net(x).reshape(n_batch, n_read, self.n_heads, 1, self.qk_dim) / np.sqrt(self.qk_dim)
            k_v = self.key_value_net(out).reshape(n_batch, n_read, self.n_heads, self.n_rules, self.qk_dim)

            comp_score = F.softmax(torch.matmul(q_v, k_v.transpose(4, 3)), dim=-1).reshape(n_batch, n_read, self.n_heads,
                                                                                           self.n_rules, 1)
        else:
            q_v = self.query_value_net(x).reshape(n_batch, n_read, self.n_heads, 1, self.qk_dim).expand(-1, -1, -1,
                                                                                                   self.n_rules, -1)
            in_ = torch.cat((q_v, out), dim=-1)
            comp_score = F.softmax(self.score_network(in_), dim=3)

        out = (comp_score * out).sum(dim=3).reshape(n_batch, n_read, self.state_size)

        return self.multi_head_merge(out), score

    def transform_data(self, input: torch.Tensor, proj: Callable[[torch.Tensor], torch.Tensor],
                       n_projs: int) -> List[torch.Tensor]:
        # Input shape: [n_batch, n_steps, n_channels]
        # Output: Tuple of n_projs tensors of dimension: [n_batch * n_heads, n_steps, projection_size]
        n_batch, n_steps, _ = input.shape
        transformed = proj(input).view(n_batch, n_steps, self.n_heads, n_projs, self.projection_size). \
            permute(0, 2, 1, 3, 4).contiguous().view(n_batch * self.n_heads, n_steps, n_projs, self.projection_size)
        return transformed.unbind(dim=2)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.multi_head_merge.weight)


class CompositionalAttention(CompositionalAttentionBase):
    def __init__(self, state_size: int, n_heads: int, n_rules: int,  qk_dim: int, dot: bool = False,
                 dropout: float = 0.1, input_size: Optional[torch.Tensor] = None):
        super().__init__(state_size, n_heads, n_rules, qk_dim, dot, dropout)

        self.query_net = torch.nn.Linear(state_size if input_size is None else input_size,
                                         n_heads * self.projection_size, bias=False)
        self.key_net = torch.nn.Linear(state_size, n_heads * self.projection_size, bias=False)
        self.value_net = torch.nn.Linear(state_size, self.projection_size * self.n_rules, bias=False)

        self.reset_parameters()

    def forward(self, curr_state: torch.Tensor, attend_to: torch.Tensor, mask: Optional[AttentionMask],
                need_weights: bool = False):
        # Input and output shape: [n_batch, n_steps, data_size]
        # No need of transform data here. Just use the default stuff.
        bsz, n_read, _ = curr_state.shape
        _, n_write, _ = attend_to.shape
        q = self.query_net(curr_state).reshape(bsz, n_read, self.n_heads, self.projection_size)
        q = q.permute(0, 2, 1, 3) * self.scale
        k = self.key_net(attend_to).reshape(bsz, n_write, self.n_heads, self.projection_size)
        k = k.permute(0, 2, 3, 1)
        v = self.value_net(attend_to).reshape(bsz, n_write, self.n_rules, self.projection_size)
        v = v.permute(0, 2, 1, 3).unsqueeze(1)

        data, scores = self.merged_attention(curr_state, mask, bsz, n_read, q, k, v)
        if need_weights:
            # Calculate the mean over the heads
            return data, scores.mean(1)
        else:
            return data

    def reset_parameters(self):
        super().reset_parameters()

        torch.nn.init.xavier_uniform_(self.query_net.weight)
        torch.nn.init.xavier_uniform_(self.key_net.weight)
        torch.nn.init.xavier_uniform_(self.value_net.weight)
