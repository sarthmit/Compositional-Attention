from itertools import product

import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.set_cmap('Greys_r')

from model import Model
from data import dataset, dataset_ood

parser = argparse.ArgumentParser(description='Toy Task for Transformer')
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--search-dim', type=int, default=64)
parser.add_argument('--value-dim', type=int, default=64)
parser.add_argument('--search', type=int, default=2)
parser.add_argument('--retrieve', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--seq-len', type=int, default=10)
parser.add_argument('--iterations', type=int, default=100000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--nonlinear', action='store_true', default=False)
parser.add_argument('--concat', action='store_true', default=False)
parser.add_argument('--no-bias', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model', type=str, default='Standard', choices=('Standard', 'Compositional-dot'))
parser.add_argument('--v-p', type=int, default=2)
parser.add_argument('--v-s', type=int, default=2)
parser.add_argument('--gumbel', action='store_true', default=False)
parser.add_argument('--separate', action='store_true', default=False)
parser.add_argument('--no-coeff', action='store_true', default=False)
parser.add_argument('--ood', action='store_true', default=False)
args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(args.seed)

if args.no_coeff:
    name = f'No_Coeff_'
else:
    name = f''

if args.ood:
    data = dataset_ood
    name = f'{name}Trained_Models/{args.seq_len}_{args.v_s}_{args.v_p}/{args.seed}'
else:
    data = dataset
    name = f'{name}Trained_Models_ood/{args.seq_len}_{args.v_s}_{args.v_p}/{args.seed}'

name = f'{name}/{args.model}_{args.dim}_{args.search_dim}_{args.value_dim}_{args.search}_{args.retrieve}'
if args.gumbel:
    name = f'{name}_gumbel'

if args.concat:
    name = f'{name}_concat'

if args.separate and args.model == 'Compositional-dot':
    name = f'{name}_separate'

if not os.path.exists(name):
    os.makedirs(name)

in_dim = args.v_s + args.v_p + args.v_s * args.v_p
all_v_combs = list(product(range(args.v_p), repeat=args.v_s))

model = Model(
    in_dim=in_dim,
    dim=args.dim,
    search_dim=args.search_dim,
    value_dim=args.value_dim,
    model=args.model,
    search=args.search,
    retrieve=args.retrieve,
    nonlinear=args.nonlinear,
    bias=not args.no_bias,
    gumbel=args.gumbel,
    concat=args.concat,
    separate=args.separate
    ).cuda()

num_params = sum(p.numel() for p in model.parameters())

print(model)
print(f"Number of Parameters: {num_params}")
with open(os.path.join(name, 'log.txt'), 'a') as f:
    f.write(f"Number of Parameters: {num_params}\n")

optimizer = Adam(model.parameters(), lr=args.lr)
criterion = nn.L1Loss()

def save(iteration, data, search, score, f_score=None):
    score = score.detach().cpu().numpy()
    if f_score is not None:
        v_score = f_score.view(-1, args.search * args.retrieve).detach().cpu().numpy()
        f_score = f_score.detach().cpu().numpy()

    if args.model == 'Compositional-dot':
        x = np.concatenate([v_score[:,:], data[:,-(args.v_p * args.v_s):]], axis=1)
        plt.imshow(x.T, vmin=0., vmax=1.)
        yticks = []
        for i in range(1,args.v_s+1):
            yticks.append(f'Ground Truth Search {i}')
            for _ in range(args.v_p - 1):
                yticks.append('')
        for i in range(1,args.search+1):
            yticks.append(f'Search {i} | Value 1')
            for j in range(2, args.retrieve+1):
                yticks.append(f'Value {j}')

        plt.yticks(ticks=np.arange(args.search * args.retrieve + args.v_s * args.v_p - 1, -1, -1), labels=yticks, rotation=45)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(name, f'iteration_{iteration}_activation.png'))
        plt.close()
    else:
        plt.imshow(data[:,-(args.v_s * args.v_p):], vmin=0., vmax=1.)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(name, f'iteration_{iteration}_task.png'))
        plt.close()

def eval_step():
    model.eval()
    total_loss = 0.

    for _ in range(100):
        if args.ood:
            data, label, _ = dataset_ood(args.batch_size, args.seq_len, args.v_s, args.v_p, all_v_combs, cff=not args.no_coeff, train=False)
        else:
            data, label, _ = dataset(args.batch_size, args.seq_len, args.v_s, args.v_p, cff=not args.no_coeff)

        data = torch.Tensor(data).cuda()
        label = torch.Tensor(label).cuda().view(-1)

        pred, _, _ = model(data)

        pred = pred.view(-1)
        loss = criterion(pred, label)

        total_loss += loss.item()

    return total_loss / 100.

def train_step():
    model.train()
    model.zero_grad()

    if args.ood:
        data, label, _ = dataset_ood(args.batch_size, args.seq_len, args.v_s, args.v_p, all_v_combs, cff=not args.no_coeff, train=True)
    else:
        data, label, _ = dataset(args.batch_size, args.seq_len, args.v_s, args.v_p, cff=not args.no_coeff)

    data = torch.Tensor(data).cuda()
    label = torch.Tensor(label).cuda().view(-1)

    pred, score, f_score = model(data)

    pred = pred.view(-1)
    loss = criterion(pred, label)

    loss.backward()
    optimizer.step()

    return loss

def plot_step(iteration):
    i = np.random.choice(args.batch_size)

    if args.ood:
        d, l, search = dataset_ood(args.batch_size, args.seq_len, args.v_s, args.v_p, all_v_combs,
                                     cff=not args.no_coeff, train=False)
    else:
        d, l, search = dataset(args.batch_size, args.seq_len, args.v_s, args.v_p, cff=not args.no_coeff)

    data = torch.Tensor(d).cuda()

    pred, score, f_score = model(data)

    save(iteration, d[i], search[i], score[i],
         f_score[i] if f_score is not None else None)

for i in range(1, args.iterations+1):
    if i % 5000 == 0 or i == 1:
        eval_loss = eval_step()
    train_loss = train_step()

    if i % 5000 == 0 or i == 1:
        log = f'Iteration: {i} | Train Loss: {train_loss:.3f}\n' \
              f'Iteration: {i} | Eval Loss: {eval_loss:.3f}\n'
        print(log)

        torch.save(model.state_dict(), os.path.join(name, 'model.pt'))

        with open(os.path.join(name, 'log.txt'), 'a') as f:
            f.write(log)

        if i % 10000 == 0:
            plot_step(i)