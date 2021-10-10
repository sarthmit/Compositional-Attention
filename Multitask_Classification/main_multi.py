
from __future__ import print_function
import argparse
import os
#import cPickle as pickle
import pickle
import random
import numpy as np
import csv

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from model import *
from dataset import TriangleDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Image Classification Transformer')
parser.add_argument('--model', type=str, choices=['Transformer', 'Compositional'], default='Transformer',
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--transformer-dim', type=int, default=128)
parser.add_argument('--qk-dim', type=int, default=16)
parser.add_argument('--iterations', default=1, type=int,
                    help='Number of Transformer Iterations to use in the relational base')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--n-rules', type=int, default=1)
parser.add_argument('--dot', action='store_true', default=False)

parser.add_argument('--name', type=str, default='Default')
args = parser.parse_args()

folder_name = f'logs/{args.name}'
model_dir = f'{folder_name}/checkpoints/'

if not os.path.isdir(folder_name):
    os.makedirs(folder_name)
    os.makedirs(model_dir)

config = {
    "Transformer Dimension": args.transformer_dim,
    "Number of Heads": args.n_heads,
    "Number of Rules": args.n_rules,
    "Iterations": args.iterations,
    "Model": args.model,
    "Seed": args.seed,
    "qk-dim": args.qk_dim,
    "dot": args.dot,
    "lr": args.lr,
}

args.cuda = not args.no_cuda
print(args)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

set_seed(args.seed)

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_f = transforms.Compose([
    transforms.Pad(2),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_ft = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

fmnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform_f)
fmnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform_ft)

svhn_train = torchvision.datasets.SVHN('./data', split='train', download=True, transform=transform)
svhn_test = torchvision.datasets.SVHN('./data', split='test', download=True, transform=transform_t)

cifar_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
cifar_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_t)

equi_train = TriangleDataset(num_examples=50000)
equi_test = TriangleDataset(num_examples=10000)


cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=args.batch_size,
                                              shuffle=True, num_workers=1)
svhn_train_loader = torch.utils.data.DataLoader(svhn_train, batch_size=args.batch_size,
                                              shuffle=True, num_workers=1)
fmnist_train_loader = torch.utils.data.DataLoader(fmnist_train, batch_size=args.batch_size,
                                              shuffle=True, num_workers=1)
equi_train_loader = torch.utils.data.DataLoader(equi_train, batch_size=args.batch_size,
                                              shuffle=True, num_workers=1)


cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=args.batch_size,
                                              shuffle=False, num_workers=1)
svhn_test_loader = torch.utils.data.DataLoader(svhn_test, batch_size=args.batch_size,
                                              shuffle=False, num_workers=1)
fmnist_test_loader = torch.utils.data.DataLoader(fmnist_test, batch_size=args.batch_size,
                                              shuffle=False, num_workers=1)
equi_test_loader = torch.utils.data.DataLoader(equi_test, batch_size=args.batch_size,
                                              shuffle=False, num_workers=1)

device = 'cuda' if args.cuda else 'cpu'

net = Model(args)
net = net.to(device)

print(net)
n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Number of Parameters: ", n_params)
with open(f'./{folder_name}/log.txt', 'w') as f:
    f.write(f"Number of Parameters: {n_params}")

bce_criterion = nn.BCEWithLogitsLoss()
ce_criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=args.lr)

def logger(results, epoch, mode='Train'):
    log = f'Epoch: {epoch}\n'
    for r in ['CIFAR10', 'FashionMNIST', 'SVHN', 'EquiTriangle']:
        log += f'{mode} | {r} | Accuracy: {results[f"{r} Accuracy"]:.3f} | Loss: {results[f"{r} Loss"]:.3f}\n'

    return log

def train():
    net.train()
    results = dict()

    correct_cifar = 0.
    correct_fmnist = 0.
    correct_svhn = 0.
    correct_equi = 0.

    total_loss_cifar = 0.
    total_loss_fmnist = 0.
    total_loss_svhn = 0.
    total_loss_equi = 0.

    total_cifar = 0.
    total_fmnist = 0.
    total_svhn = 0.
    total_equi = 0.

    cifar_iterator = iter(cifar_train_loader)
    fmnist_iterator = iter(fmnist_train_loader)
    svhn_iterator = iter(svhn_train_loader)
    equi_iterator = iter(equi_train_loader)

    for batch_idx, (s_inp, s_tar) in enumerate(svhn_iterator):

        try:
            c_inp, c_tar = next(cifar_iterator)
        except StopIteration:
            cifar_iterator = iter(cifar_train_loader)
            c_inp, c_tar = next(cifar_iterator)

        try:
            f_inp, f_tar = next(fmnist_iterator)
        except StopIteration:
            fmnist_iterator = iter(fmnist_train_loader)
            f_inp, f_tar = next(fmnist_iterator)

        try:
            e_inp, e_tar = next(equi_iterator)
        except StopIteration:
            equi_iterator = iter(equi_train_loader)
            e_inp, e_tar = next(equi_iterator)

        f_inp = f_inp.repeat(1, 3, 1, 1)
        e_inp = e_inp.repeat(1, 3, 1, 1)
        e_inp = (e_inp - 127.5) / 127.5

        c_inp, f_inp, s_inp, e_inp = c_inp.to(device), f_inp.to(device), s_inp.to(device), e_inp.to(device)
        c_tar, f_tar, s_tar, e_tar = c_tar.to(device), f_tar.to(device), s_tar.to(device), e_tar.to(device)
        e_tar = e_tar.float()

        out_c, _, _, _ = net(c_inp)
        _, out_f, _, _ = net(f_inp)
        _, _, out_s, _ = net(s_inp)
        _, _, _, out_e = net(e_inp)

        loss_c = ce_criterion(out_c, c_tar)
        loss_f = ce_criterion(out_f, f_tar)
        loss_s = ce_criterion(out_s, s_tar)
        loss_e = bce_criterion(out_e, e_tar)

        _, pred_c = out_c.max(dim = 1)
        _, pred_f = out_f.max(dim = 1)
        _, pred_s = out_s.max(dim = 1)
        pred_e = (torch.sigmoid(out_e) >= 0.5).int()

        optimizer.zero_grad()
        (loss_c + loss_f + loss_s + loss_e).backward()
        optimizer.step()

        total_loss_cifar += loss_c.item() * c_tar.size(0)
        total_loss_fmnist += loss_f.item() * f_tar.size(0)
        total_loss_svhn += loss_s.item() * s_tar.size(0)
        total_loss_equi += loss_e.item() * e_tar.size(0)

        total_cifar += c_tar.size(0)
        total_fmnist += f_tar.size(0)
        total_svhn += s_tar.size(0)
        total_equi += e_tar.size(0)

        correct_cifar += torch.eq(pred_c, c_tar).sum().item()
        correct_fmnist += torch.eq(pred_f, f_tar).sum().item()
        correct_svhn += torch.eq(pred_s, s_tar).sum().item()
        correct_equi += torch.eq(pred_e, e_tar).sum().item()

    loss_c = total_loss_cifar / total_cifar
    acc_c = correct_cifar / total_cifar * 100
    loss_f = total_loss_fmnist / total_fmnist
    acc_f = correct_fmnist / total_fmnist * 100
    loss_s = total_loss_svhn / total_svhn
    acc_s = correct_svhn / total_svhn * 100
    loss_e = total_loss_equi / total_equi
    acc_e = correct_equi / total_equi * 100

    results['CIFAR10 Accuracy'] = acc_c
    results['FashionMNIST Accuracy'] = acc_f
    results['SVHN Accuracy'] = acc_s
    results['EquiTriangle Accuracy'] = acc_e

    results['CIFAR10 Loss'] = loss_c
    results['FashionMNIST Loss'] = loss_f
    results['SVHN Loss'] = loss_s
    results['EquiTriangle Loss'] = loss_e

    return results

def test(dataset, results):
    net.eval()

    if dataset == 'EquiTriangle':
        test_loader = equi_test_loader
        criterion = bce_criterion
    elif dataset == 'CIFAR10':
        test_loader = cifar_test_loader
        criterion = ce_criterion
    elif dataset == 'FashionMNIST':
        test_loader = fmnist_test_loader
        criterion = ce_criterion
    elif dataset == 'SVHN':
        test_loader = svhn_test_loader
        criterion = ce_criterion
    else:
        print("Dataset not found.")

    correct = 0.
    total_loss = 0.
    total = 0.

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            if dataset == "EquiTriangle":
                inputs = (inputs - 127.5) / 127.5

            if dataset == 'EquiTriangle' or dataset == 'FashionMNIST':
                inputs = inputs.repeat(1, 3, 1, 1)

            out_c, out_f, out_s, out_e = net(inputs)

            if dataset == 'CIFAR10':
                out = out_c
                _, prediction = out.max(dim=1)
            elif dataset == 'FashionMNIST':
                out = out_f
                _, prediction = out.max(dim=1)
            elif dataset == 'SVHN':
                out = out_s
                _, prediction = out.max(dim=1)
            elif dataset == 'EquiTriangle':
                out = out_e
                prediction = (torch.sigmoid(out) >= 0.5).int()
                targets = targets.float()

            loss = criterion(out, targets)
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            correct += torch.eq(prediction, targets).sum().item()

    loss = total_loss / total
    accuracy = correct / total

    results[f'{dataset} Accuracy'] = accuracy * 100
    results[f'{dataset} Loss'] = loss

def eval():
    results = dict()
    test("EquiTriangle", results)
    test("CIFAR10", results)
    test("SVHN", results)
    test("FashionMNIST", results)

    return results

results = eval()
log = logger(results, 0, 'Eval')
with open(f'./{folder_name}/log.txt', 'a') as f:
    f.write(log)
print(log)

for epoch in range(1, args.epochs + 1):
    train_results = train()
    log = logger(train_results, epoch, 'Train')

    with open(f'./{folder_name}/log.txt', 'a') as f:
        f.write(log)
    print(log)

    eval_results = eval()
    log = logger(eval_results, epoch, 'Eval')
    with open(f'./{folder_name}/log.txt', 'a') as f:
        f.write(log)
    print(log)

    final_dict = dict()
    for key in train_results.keys():
        final_dict[f'Train {key}'] = train_results[f'{key}']
        final_dict[f'Eval {key}'] = eval_results[f'{key}']

    torch.save(net.state_dict(), os.path.join(f'{folder_name}/', 'model.pt'))