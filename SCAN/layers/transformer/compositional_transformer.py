import torch
import torch.nn
import torch.nn.functional as F
from .compositional_attention import CompositionalAttention, AttentionMask
from typing import Optional, Callable, Dict
from dataclasses import dataclass

# This file is based on PyTorch's internal implementation

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


class CompTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, nrules, qk_dim,  dim_feedforward=2048, dropout=0.1,
                 activation: ActivationFunction = F.relu):
        super(CompTransformerEncoderLayer, self).__init__()
        self.self_attn = CompositionalAttention(d_model, nhead, nrules, qk_dim, dropout=dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, AttentionMask(mask, None))
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu')
        if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)


class CompTransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, nrules, qk_dim, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu):
        super(CompTransformerDecoderLayer, self).__init__()

        self.self_attn = CompositionalAttention(d_model, nhead, nrules, qk_dim, dropout=dropout)
        self.multihead_attn = CompositionalAttention(d_model, nhead, nrules, qk_dim, dropout=dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                full_target: Optional[torch.Tensor] = None, pos_offset: int = 0) -> torch.Tensor:
        assert pos_offset == 0 or tgt_mask is None
        tgt2 = self.self_attn(tgt, tgt if full_target is None else full_target, mask=AttentionMask(None, tgt_mask))
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, mask=AttentionMask(memory_key_padding_mask, None))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu')
        if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)


class CompositionalTransformerDecoderBase(torch.nn.Module):
    @dataclass
    class State:
        step: int
        state: Dict[int, torch.Tensor]

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def create_state(self, batch_size: int, max_length: int, device: torch.device) -> State:
        return self.State(0, {i: torch.empty([batch_size, max_length, self.d_model], device=device)
                              for i in range(len(self.layers))})

    def one_step_forward(self, state: State, data: torch.Tensor, *args, **kwargs):
        assert data.shape[1] == 1, f"For one-step forward should have one timesteps, but shape is {data.shape}"
        assert state.step < state.state[0].shape[1]

        for i, l in enumerate(self.layers):
            state.state[i][:, state.step:state.step + 1] = data
            data = l(data, *args, **kwargs, full_target=state.state[i][:, :state.step + 1],
                     pos_offset=state.step)

        state.step += 1
        return data


class CompTransformerEncoder(torch.nn.Module):
    def __init__(self, layer, depth: int, *args, **kwargs):
        super().__init__()
        self.layer = layer(*args, **kwargs)
        self.layers = [self.layer] * depth

    def forward(self, data: torch.Tensor, *args, **kwargs):
        for l in self.layers:
            data = l(data, *args, **kwargs)
        return data


class CompTransformerDecoder(CompositionalTransformerDecoderBase):
    def __init__(self, layer, depth: int, d_model: int, *args, **kwargs):
        super().__init__(d_model)
        self.layer = layer(d_model, *args, **kwargs)
        self.layers = [self.layer] * depth

    def forward(self, data: torch.Tensor, *args, **kwargs):
        for l in self.layers:
            data = l(data, *args, **kwargs)
        return data


def CompTransformerEncoderWithLayer(layer=CompTransformerEncoderLayer):
    return lambda *args, **kwargs: CompTransformerEncoder(layer, *args, **kwargs)


def CompTransformerDecoderWithLayer(layer=CompTransformerDecoderLayer):
    return lambda *args, **kwargs: CompTransformerDecoder(layer, *args, **kwargs)


class CompositionalTransformer(torch.nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, nrules: int = 4, qk_dim: int = 4, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: ActivationFunction = F.relu, encoder_layer=CompTransformerEncoderWithLayer(),
                 decoder_layer=CompTransformerDecoderWithLayer()):
        super().__init__()

        self.encoder = encoder_layer(num_encoder_layers, d_model, nhead, nrules, qk_dim, dim_feedforward,
                                     dropout, activation)
        self.decoder = decoder_layer(num_decoder_layers, d_model, nhead, nrules, qk_dim, dim_feedforward,
                                     dropout, activation)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                src_length_mask: Optional[torch.Tensor] = None):
        memory = self.encoder(src, src_length_mask)
        return self.decoder(tgt, memory, tgt_mask, src_length_mask)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

