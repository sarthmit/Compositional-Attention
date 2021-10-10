
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

def get_positional(seq_len, dim):
    pe = torch.zeros(seq_len, dim)
    normalizer = 1. / (1. + math.exp(-1))
    for pos in range(seq_len):
        for i in range(0, dim, 2):
            pe[pos, i] = normalizer * math.sin(pos / (10000 ** ((2 * i)/dim)))
            pe[pos, i+1] = normalizer * math.cos(pos / (10000 ** ((2 * (i+1))/dim)))

    pe = pe.unsqueeze(0)
    return pe

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss
        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def save_model(self, epoch, name):
        torch.save(self.state_dict(), f'{name}/epoch_{epoch:02d}.pth')

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
        bsz, n_read, _ = x.shape
        _, n_write, _ = x.shape

        res = x
        if self.norm_before:
            x = self.norm1(x)

        q = self.query_net(x).reshape(bsz, n_read, self.nheads, self.head_dim)
        q = q.permute(0,2,1,3) / np.sqrt(self.head_dim)
        k = self.key_net(x).reshape(bsz, n_write, self.nheads, self.head_dim)
        k = k.permute(0,2,3,1)
        v = self.value_net(x).reshape(bsz, n_write, self.nheads, self.head_dim)
        v = v.permute(0,2,1,3)

        score = F.softmax(torch.matmul(q,k), dim=-1) # (bsz, nheads, n_read, n_write)

        out = torch.matmul(score, v) # (bsz, nheads, n_read, att_dim)
        out = out.view(bsz, self.nheads, n_read, self.head_dim)

        out = out.permute(0, 2, 1, 3).reshape(bsz, n_read, self.dim)
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

class Compositional_Self_Attention(nn.Module):
    def __init__(self, dim, qk_dim=16, nheads=4, nrules=1, dot=False):
        super(Compositional_Self_Attention, self).__init__()

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

    def forward(self, x):
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

            # GUMBEL OR NOT?
            comp_score = F.softmax(torch.matmul(q_v, k_v.transpose(4,3)), dim=-1).reshape(bsz, n_read, self.nheads, self.nrules, 1)
        else:
            q_v = self.query_value_net(x).reshape(bsz, n_read, self.nheads, 1, self.qk_dim).expand(-1, -1, -1, self.nrules, -1)
            in_ = torch.cat((q_v, out), dim=-1)

            # GUMBEL OR NOT?
            comp_score = F.softmax(self.score_network(in_), dim=3)

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

        return out

class Encoder(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, question_size, in_channels=3, hidden_size=256):
        super(Encoder, self).__init__()
        img_size = _pair(img_size)

        patch_size = (15, 15)
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.question_representation = nn.Linear(question_size, hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.pe = get_positional(25, hidden_size).cuda()

    def forward(self, x, que):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        que = self.question_representation(que).unsqueeze(1).expand(-1, self.n_patches+1, -1)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = x + self.pe
        embeddings = torch.cat((cls_tokens, x), dim=1)
        embeddings = torch.cat((embeddings, que), dim=-1)
        return embeddings

class Model(BasicModel):
    def __init__(self, args):
        super(Model, self).__init__(args, 'Model')

        self.transformer_dim = args.transformer_dim
        self.qk_dim = args.qk_dim
        self.heads = args.n_heads
        self.rules = args.n_rules
        self.relation_type = args.relation_type
        self.iterations = args.iterations
        self.dot = args.dot

        self.encoder = Encoder((75, 75), 18, hidden_size = self.transformer_dim // 2)
        self.mapping = nn.Linear((self.transformer_dim // 2) * 2, args.transformer_dim)

        if args.model == 'Transformer':
            self.transformer = Self_Attention(self.transformer_dim, self.heads)
        elif args.model == 'Compositional':
            self.transformer = Compositional_Self_Attention(self.transformer_dim, self.qk_dim, self.heads, self.rules, self.dot)

        self.final = nn.Linear(self.transformer_dim, 10)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, img, qst):
        x = self.encoder(img, qst)
        x = self.mapping(x)

        for _ in range(self.iterations):
            x = self.transformer(x)

        y = self.final(x[:,0,:])
        return F.log_softmax(y, dim=1)

if __name__=="__main__":
    image = torch.randn(1, 3, 75, 75)
    question = torch.randn(1, 18)
    embedding = Encoder(image.shape[-2:], question.shape[-1])
    out = embedding(image, question)
    print(out.shape)
