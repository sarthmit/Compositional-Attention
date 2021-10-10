import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

class GroupLinearLayer(nn.Module):
    """Modularized Linear Layer"""
    def __init__(self, num_blocks, din, dout, bias=True):
        super(GroupLinearLayer, self).__init__()

        self.bias=bias
        self.w = nn.Parameter(torch.Tensor(num_blocks, din, dout))
        self.b = nn.Parameter(torch.Tensor(1, num_blocks, dout))

        stdv = math.sqrt(6.0) / math.sqrt(din + dout)
        nn.init.uniform_(self.w, -stdv, stdv)
        nn.init.zeros_(self.b)

    def forward(self,x):
        # x - (bsz, num_blocks, din)
        x = x.permute(1,0,2)
        x = torch.bmm(x, self.w)
        x = x.permute(1,0,2)

        if self.bias:
            x = x + self.b

        return x

class Compositional_dot_Transformer(nn.Module):
    def __init__(self, dim, search_dim, value_dim, search, retrieval, nonlinear, gumbel, concat, separate, bias):
        super(Compositional_dot_Transformer, self).__init__()

        self.dim = dim
        self.search_dim = search_dim
        self.value_dim = value_dim
        self.head_dim = search_dim // search
        self.head_v_dim = value_dim // retrieval
        self.nonlinear = nonlinear
        self.search = search
        self.retrieval = retrieval
        self.scaling = self.head_dim ** -0.5
        self.gumbel = gumbel
        self.concat = concat
        self.separate = separate

        self.query_net = nn.Linear(dim, search_dim, bias=bias)
        self.key_net = nn.Linear(dim, search_dim, bias=bias)
        self.value_net = nn.Linear(dim, value_dim, bias=bias)

        assert(self.head_dim * search == search_dim)
        assert(self.head_v_dim * retrieval == value_dim)

        self.value_query = nn.Linear(dim, search_dim, bias=bias)
        if self.separate:
            self.value_key = GroupLinearLayer(self.retrieval, self.head_v_dim, self.head_dim, bias=bias)
        else:
            self.value_key = nn.Linear(self.head_v_dim, self.head_dim, bias=bias)

        extra = 0
        if self.concat:
            self.in_ = nn.Linear(dim, self.head_v_dim, bias=bias)
            extra = 1

        if self.nonlinear:
            self.out_proj = nn.Sequential(
                nn.Linear((self.search + extra) * self.head_v_dim, dim, bias=bias),
                nn.ReLU(),
                nn.Linear(dim, dim, bias=bias)
            )
        else:
            self.out_proj = nn.Linear((self.search + extra) * self.head_v_dim, dim, bias=bias)

    def forward(self, x):
        bsz, n, _ = x.shape

        q = self.query_net(x).view(bsz, n, self.search, self.head_dim) * self.scaling
        k = self.key_net(x).view(bsz, n, self.search, self.head_dim)
        v = self.value_net(x).view(bsz, n, self.retrieval, self.head_v_dim)

        q = q.transpose(2,1).contiguous()
        k = k.permute(0, 2, 3, 1).contiguous()
        v = v.transpose(2,1).contiguous().unsqueeze(1) # (bsz, 1, retrieval, n, head_v_dim)

        score = torch.matmul(q, k) # (bsz, search, n, n)
        mask = torch.zeros_like(score[0,0]).fill_diagonal_(1).unsqueeze(0).unsqueeze(0)
        mask = mask.repeat(bsz, self.search, 1, 1).bool()
        score.masked_fill_(mask, float('-inf'))

        if self.gumbel:
            score = F.gumbel_softmax(score, dim=-1).unsqueeze(2)
        else:
            score = F.softmax(score, dim=-1).unsqueeze(2) # (bsz, search, 1, n, n)

        out = torch.matmul(score, v).permute(0, 3, 1, 2, 4).reshape(bsz, n, self.search, self.retrieval, self.head_v_dim)
        q_v = self.value_query(x).view(bsz, n, self.search, 1, self.head_dim) * self.scaling

        if self.separate:
            z = out.contiguous().view(bsz * n * self.search, self.retrieval, self.head_v_dim)
            k_v = self.value_key(z).view(bsz, n, self.search, self.retrieval, self.head_v_dim)
        else:
            k_v = self.value_key(out)

        k_v = k_v.permute(0, 1, 2, 4, 3).contiguous() # (bsz, n, search, head_dim, retrieval)
        v_score = torch.matmul(q_v, k_v).view(bsz, n, self.search, self.retrieval, 1)

        if self.gumbel:
            v_score = F.gumbel_softmax(v_score, dim=3) # (bsz, n, search, retrieval, 1)
        else:
            v_score = F.softmax(v_score, dim=3) # (bsz, n, search, retrieval, 1)

        out = (v_score * out).sum(dim=3).reshape(bsz, n, self.search * self.head_v_dim)

        if self.concat:
            in_ = self.in_(x)
            out = torch.cat([out, in_], dim=-1)

        return self.out_proj(out), score, v_score

class Transformer(nn.Module):
    def __init__(self, dim, search_dim, value_dim, search, retrieve, nonlinear, gumbel, concat, bias):
        super(Transformer, self).__init__()

        self.dim = dim
        self.search_dim = search_dim
        self.value_dim = value_dim

        assert(search == retrieve)

        self.head_dim = search_dim // search
        self.head_v_dim = value_dim // search
        self.search = search
        self.nonlinear = nonlinear
        self.scaling = self.head_dim ** -0.5
        self.gumbel = gumbel
        self.concat = concat

        self.query_net = nn.Linear(dim, search_dim, bias=bias)
        self.key_net = nn.Linear(dim, search_dim, bias=bias)
        self.value_net = nn.Linear(dim, value_dim, bias=bias)

        assert(search * self.head_dim == search_dim)
        assert(search * self.head_v_dim == value_dim)

        extra = 0
        if concat:
            self.in_ = nn.Linear(dim, self.head_v_dim, bias=bias)
            extra = 1

        if self.nonlinear:
            self.out_proj = nn.Sequential(
                nn.Linear((self.search + extra) * self.head_v_dim, dim, bias=bias),
                nn.ReLU(),
                nn.Linear(dim, dim, bias=bias)
            )
        else:
            self.out_proj = nn.Linear((self.search + extra) * self.head_v_dim, dim, bias=bias)

    def forward(self, x):
        bsz, n, _ = x.shape

        q = self.query_net(x).view(bsz, n, self.search, self.head_dim) * self.scaling
        k = self.key_net(x).view(bsz, n, self.search, self.head_dim)
        v = self.value_net(x).view(bsz, n, self.search, self.head_v_dim)

        q = q.transpose(2,1).contiguous()
        k = k.permute(0, 2, 3, 1).contiguous()
        v = v.transpose(2,1).contiguous()

        score = torch.matmul(q, k) # (bsz, search, n, n)
        mask = torch.zeros_like(score[0,0]).fill_diagonal_(1).unsqueeze(0).unsqueeze(0)
        mask = mask.repeat(bsz, self.search, 1, 1).bool()
        score.masked_fill_(mask, float('-inf'))
        if self.gumbel:
            score = F.gumbel_softmax(score, dim=-1)
        else:
            score = F.softmax(score, dim=-1)

        out = torch.matmul(score, v).transpose(2, 1).reshape(bsz, n, self.search * self.head_v_dim)
        if self.concat:
            in_ = self.in_(x)
            out = torch.cat([out, in_], dim=-1)

        return self.out_proj(out), score, None

class Model(nn.Module):
    def __init__(self, in_dim=3, dim=64, search_dim=64, value_dim=64,
                 search=4, retrieve=4, model='Standard',
                 nonlinear=False, concat=False,
                 bias=True, separate=False, gumbel=False):
        super(Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        if model == 'Compositional-dot':
            self.model = Compositional_dot_Transformer(dim, search_dim, value_dim, search, retrieve, nonlinear, gumbel, concat, separate, bias)
        elif model == 'Standard':
            self.model = Transformer(dim, search_dim, value_dim, search, retrieve, nonlinear, gumbel, concat, bias)
        else:
            print("Incorrect Model Specified")
            exit()

    def forward(self, x):
        x = self.encoder(x)
        x, score, f_score = self.model(x)
        x = self.decoder(x)

        return x, score, f_score