
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair

def get_positional(seq_len, dim):
    pe = torch.zeros(seq_len, dim)
    normalizer = 1. / (1. + math.exp(-1))
    for pos in range(seq_len):
        for i in range(0, dim, 2):
            pe[pos, i] = normalizer * math.sin(pos / (10000 ** ((2 * i)/dim)))
            pe[pos, i+1] = normalizer * math.cos(pos / (10000 ** ((2 * (i+1))/dim)))

    pe = pe.unsqueeze(0)
    return pe

class Self_Attention(nn.Module):
    def __init__(self, dim, nheads=4):
        super(Self_Attention, self).__init__()

        self.dim = dim
        self.nheads = nheads
        self.head_dim = dim // nheads

        self.norm_before = True

        self.query_net = nn.Linear(dim, dim)
        self.key_net = nn.Linear(dim, dim)
        self.value_net = nn.Linear(dim, dim)

        self.final = nn.Linear(dim, dim)

        self.res = nn.Sequential(
            nn.Linear(dim,2 * dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
            nn.Dropout(p=0.1)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        bsz, n, _ = x.shape

        res = x
        if self.norm_before:
            x = self.norm1(x)

        q = self.query_net(x).reshape(bsz, n, self.nheads, self.head_dim)
        q = q.permute(0,2,1,3) / np.sqrt(self.head_dim)
        k = self.key_net(x).reshape(bsz, n, self.nheads, self.head_dim)
        k = k.permute(0,2,3,1)
        v = self.value_net(x).reshape(bsz, n, self.nheads, self.head_dim)
        v = v.permute(0,2,1,3)

        score = F.softmax(torch.matmul(q,k), dim=-1) # (bsz, nheads, n, n)

        out = torch.matmul(score, v) # (bsz, nheads, n, att_dim)
        out = out.view(bsz, self.nheads, n, self.head_dim)

        out = out.permute(0, 2, 1, 3).reshape(bsz, n, self.dim)
        out = self.final(out)

        if not self.norm_before:
            out = self.norm1(res + out)
        else:
            out = res + out

        res = out

        if self.norm_before:
            out = self.norm2(out)
            out = res + self.res(out)
        else:
            out = self.norm2(res + self.res(out))

        return out

class Compositional_Attention(nn.Module):
    def __init__(self, dim, qk_dim=16, nheads=4, nrules=1, dot=False):
        super(Compositional_Attention, self).__init__()

        self.dim = dim
        self.nheads = nheads
        self.nrules = nrules
        self.head_dim = dim // nheads
        self.qk_dim = qk_dim
        self.dot = dot

        self.norm_before = True

        self.query_net = nn.Linear(dim, dim)
        self.key_net = nn.Linear(dim, dim)
        self.value_net = nn.Linear(dim, self.head_dim * self.nrules)

        self.query_value_net = nn.Linear(dim, self.qk_dim * nheads)

        if dot:
            self.key_value_net = nn.Linear(self.head_dim, self.qk_dim)
        else:
            self.score_network = nn.Linear(self.head_dim + self.qk_dim, 1)

        self.final = nn.Linear(dim, dim)

        self.res = nn.Sequential(
            nn.Linear(dim,2 * dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
            nn.Dropout(p=0.1)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, vis=False):
        bsz, n_read, _ = x.shape
        _, n_write, _ = x.shape

        res = x
        if self.norm_before:
            x = self.norm1(x)

        q = self.query_net(x).reshape(bsz, n_read, self.nheads, self.head_dim)
        q = q.permute(0,2,1,3) / np.sqrt(self.head_dim)
        k = self.key_net(x).reshape(bsz, n_write, self.nheads, self.head_dim)
        k = k.permute(0,2,3,1)
        v = self.value_net(x).reshape(bsz, n_write, self.nrules, self.head_dim)
        v = v.permute(0,2,1,3).unsqueeze(1)

        score = F.softmax(torch.matmul(q,k), dim=-1).unsqueeze(2) # (bsz, nheads, n_read, n_write)

        out = torch.matmul(score, v) # (bsz, nheads, nrules, n_read, att_dim)
        out = out.view(bsz, self.nheads, self.nrules, n_read, self.head_dim)

        out = out.permute(0, 3, 1, 2, 4).reshape(bsz, n_read, self.nheads, self.nrules, self.head_dim)

        if self.dot:
            q_v = self.query_value_net(x).reshape(bsz, n_read, self.nheads, 1, self.qk_dim) / np.sqrt(self.qk_dim)
            k_v = self.key_value_net(out).reshape(bsz, n_read, self.nheads, self.nrules, self.qk_dim)

            comp_score = torch.matmul(q_v, k_v.transpose(4,3))
        else:
            q_v = self.query_value_net(x).reshape(bsz, n_read, self.nheads, 1, self.qk_dim).expand(-1, -1, -1, self.nrules, -1)
            in_ = torch.cat((q_v, out), dim=-1)
            comp_score = self.score_network(in_)

        comp_score = comp_score.reshape(bsz, n_read, self.nheads, self.nrules, 1)
        comp_score = F.softmax(comp_score, dim=3)

        out = (comp_score * out).sum(dim=3).reshape(bsz, n_read, self.dim)

        out = self.final(out)

        if not self.norm_before:
            out = self.norm1(res + out)
        else:
            out = res + out

        res = out

        if self.norm_before:
            out = self.norm2(out)
            out = res + self.res(out)
        else:
            out = self.norm2(res + self.res(out))

        if vis:
            return out, comp_score

        return out

class Encoder(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, channels=3, hidden_size=256):
        super(Encoder, self).__init__()
        img_size = _pair(img_size)

        self.patch_size = (4, 4)
        self.n_horiz = img_size[0] // self.patch_size[0]
        self.n_vert = img_size[1] // self.patch_size[1]

        self.n_patches = self.n_horiz * self.n_vert

        self.embeddings = nn.Sequential(
            nn.Linear(channels * self.patch_size[0] * self.patch_size[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 4, channels * self.patch_size[0] * self.patch_size[1]))
        self.pe = get_positional(self.n_patches, hidden_size).cuda()

    def forward(self, x):
        # x - B, C, H, W

        B,C,H,W = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = x.reshape(B, C, self.n_horiz, self.patch_size[0], self.n_vert, self.patch_size[1])
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, self.n_patches, C * self.patch_size[0] * self.patch_size[1])

        x = self.embeddings(x) + self.pe
        cls_tokens = self.embeddings(cls_tokens)

        return torch.cat([cls_tokens, x], dim=1)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.transformer_dim = args.transformer_dim
        self.qk_dim = args.qk_dim
        self.heads = args.n_heads
        self.rules = args.n_rules
        self.iterations = args.iterations
        self.dot = args.dot
        self.model = args.model

        img_size = (32, 32)
        channels = 3

        self.encoder = Encoder(img_size, channels = channels, hidden_size = self.transformer_dim)

        if args.model == 'Transformer':
            self.transformer = Self_Attention(self.transformer_dim, self.heads)
        elif args.model == 'Compositional':
            self.transformer = Compositional_Attention(self.transformer_dim, self.qk_dim, self.heads, self.rules, self.dot)

        self.out_layer = nn.Sequential(
            nn.Linear(self.transformer_dim, 128),
            nn.ReLU()
        )

        self.fmnist_layer = nn.Linear(128, 10)
        self.cifar_layer = nn.Linear(128, 10)
        self.svhn_layer = nn.Linear(128, 10)
        self.eq_layer = nn.Linear(128, 1)

    def forward(self, img):
        x = self.encoder(img)

        for _ in range(self.iterations):
            x = self.transformer(x)

        y = self.out_layer(x[:,:4,:])

        return self.cifar_layer(y[:, 0, :]), self.fmnist_layer(y[:, 1, :]), self.svhn_layer(y[:, 2, :]), self.eq_layer(y[:, 3, :]).squeeze()