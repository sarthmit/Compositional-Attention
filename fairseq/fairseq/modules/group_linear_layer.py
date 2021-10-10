import torch
import torch.nn as nn
import math

class GroupLinearLayer(nn.Module):
    """Modularized Linear Layer"""
    def __init__(self, num_blocks, din, dout, bias=True):
        super(GroupLinearLayer, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(num_blocks, din, dout))
        self.bias = bias
        self.din = din
        self.dout = dout
        self.num_blocks = num_blocks
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, num_blocks, dout))
        else:
            self.bias = None

    def reset_params(self):
        stdv = math.sqrt(3.0) * math.sqrt(2.0 / (self.din + self.dout))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self):
        return 'num_blocks={}, in_features={}, out_features={}, bias={}'.format(
            self.num_blocks, self.din, self.dout, self.bias is not None
        )

    def forward(self, x):
        # x - (tgt_len, bsz, num_blocks, din)
        tgt_len, bsz, _, _ = x.shape
        x = (x
            .view(tgt_len * bsz, self.num_blocks, self.din)
            .permute(1,0,2)
        )
        x = torch.bmm(x, self.weight)
        x = (x
             .permute(1,0,2)
             .reshape(tgt_len, bsz, self.num_blocks, self.dout)
        )

        if self.bias is not None:
            x = x + self.bias

        return x
