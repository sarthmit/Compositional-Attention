import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from util import log
import numpy as np
from modules import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Compositional_Self_Attention(nn.Module):
    def __init__(self, dim, ffn_dim, nheads=4, nrules=1, qk_dim=16, gumbel=False):
        super(Compositional_Self_Attention, self).__init__()

        self.dim = dim
        self.ffn_dim = ffn_dim
        self.nheads = nheads
        self.nrules = nrules
        self.head_dim = dim // nheads
        self.qk_dim = qk_dim
        self.gumbel = gumbel

        self.norm_before = False

        self.query_net = nn.Linear(dim, dim)
        self.key_net = nn.Linear(dim, dim)
        self.value_net = nn.Linear(dim, self.head_dim * self.nrules)

        self.query_value_net = nn.Linear(dim, self.qk_dim * nheads)
        self.key_value_net = nn.Linear(self.head_dim, self.qk_dim)

        self.final = nn.Linear(dim, dim)

        self.res = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.Dropout(p=0.0),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(p=0.0)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.transpose(1,0).contiguous()
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

        q_v = self.query_value_net(x).reshape(bsz, n_read, self.nheads, 1, self.qk_dim) / np.sqrt(self.qk_dim)
        k_v = self.key_value_net(out).reshape(bsz, n_read, self.nheads, self.nrules, self.qk_dim)

        # GUMBEL OR NOT?
        if self.gumbel:
            comp_score = F.gumbel_softmax(torch.matmul(q_v, k_v.transpose(4, 3)), dim=-1).reshape(bsz, n_read, self.nheads, self.nrules, 1)
        else:
            comp_score = F.softmax(torch.matmul(q_v, k_v.transpose(4,3)), dim=-1).reshape(bsz, n_read, self.nheads, self.nrules, 1)

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

        return out.transpose(1,0).contiguous()

class Model(nn.Module):
    def __init__(self, task_gen, args):
        super(Model, self).__init__()
        # Encoder
        log.info('Building encoder...')
        if args.encoder == 'conv':
            self.encoder = Encoder_conv(args)
        elif args.encoder == 'mlp':
            self.encoder = Encoder_mlp(args)
        elif args.encoder == 'rand':
            self.encoder = Encoder_rand(args)
        self.z_size = 128

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.z_size)

        # Transformer
        log.info('Building transformer encoder...')
        self.separate = args.separate
        self.n_heads = args.heads
        self.n_rules = args.rules
        self.dim_ff = 512
        self.n_layers = args.layers
        self.gumbel = args.gumbel
        self.encoder_norm = nn.LayerNorm(self.z_size)
        self.transformer_encoder = Compositional_Self_Attention(self.z_size, self.dim_ff, self.n_heads, self.n_rules, gumbel=self.gumbel)

        if self.separate:
            self.out_token = nn.Parameter(torch.randn(1, 1, self.z_size))
            nn.init.xavier_normal_(self.out_token)

        # Output layers
        log.info('Building output layers...')
        self.out_hidden = nn.Linear(self.z_size, 256)
        self.y_out = nn.Linear(256, task_gen.y_dim)

        # Context normalization
        if args.norm_type == 'contextnorm' or args.norm_type == 'tasksegmented_contextnorm':
            self.contextnorm = True
            self.gamma = nn.Parameter(torch.ones(self.z_size))
            self.beta = nn.Parameter(torch.zeros(self.z_size))
        else:
            self.contextnorm = False
        if args.norm_type == 'tasksegmented_contextnorm':
            self.task_seg = task_gen.task_seg
        else:
            self.task_seg = [np.arange(task_gen.seq_len)]

        # Nonlinearities
        self.relu = nn.ReLU()

        # Initialize parameters
        for name, param in self.named_parameters():
            # Initialize all biases to 0
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                # Initialize transformer parameters
                if 'transformer' in name:
                    # Initialize attention weights using Xavier normal distribution
                    if 'net' in name:
                        nn.init.xavier_normal_(param, gain=1./math.sqrt(2))
                    elif 'final' in name:
                        nn.init.xavier_normal_(param)
                    # Initialize feedforward weights (followed by ReLU) using Kaiming normal distribution
                    if 'res' in name:
                        nn.init.kaiming_normal_(param, nonlinearity='relu')
                # Initialize output layers
                # Initialize output hidden layer (followed by ReLU) using Kaiming normal distribution
                if 'out_hidden' in name:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                # Initialize weights for multiple-choice output layer (followed by softmax) using Xavier normal distribution
                if 'y_out' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_seq, device):
        # Encode all images in sequence
        z_seq = []
        for t in range(x_seq.shape[1]):
            x_t = x_seq[:,t,:,:].unsqueeze(1)
            z_t = self.encoder(x_t)
            z_seq.append(z_t)
        z_seq = torch.stack(z_seq, dim=0)

        if self.contextnorm:
            z_seq_all_seg = []
            for seg in range(len(self.task_seg)):
                z_seq_all_seg.append(self.apply_context_norm(z_seq[self.task_seg[seg],:,:]))
            z_seq = torch.cat(z_seq_all_seg, dim=0)

        # Positional encoding
        z_seq_pe = self.pos_encoder(z_seq)
        if self.separate:
            z_seq_pe = torch.cat([z_seq_pe, self.out_token.repeat(1, z_seq_pe.shape[1], 1)], dim=0)

        # Apply transformer
        z_seq_transformed = z_seq_pe
        for _ in range(self.n_layers):
            z_seq_transformed = self.encoder_norm(self.transformer_encoder(z_seq_transformed))

        # Average over transformed sequence
        if self.separate:
            z_seq_transformed_avg = z_seq_transformed[-1, :, :]
        else:
            z_seq_transformed_avg = z_seq_transformed.mean(0)

        # Output layers
        out_hidden = self.relu(self.out_hidden(z_seq_transformed_avg))
        y_pred_linear = self.y_out(out_hidden)
        y_pred = y_pred_linear.argmax(1)

        return y_pred_linear, y_pred

    def apply_context_norm(self, z_seq):
        eps = 1e-8
        z_mu = z_seq.mean(0)
        z_sigma = (z_seq.var(0) + eps).sqrt()
        z_seq = (z_seq - z_mu.unsqueeze(0)) / z_sigma.unsqueeze(0)
        z_seq = (z_seq * self.gamma) + self.beta
        return z_seq