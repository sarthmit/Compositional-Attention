"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
from __future__ import print_function
import argparse
import os

import pickle
import random
import numpy as np
import csv

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from model import *
import wandb

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['Transformer', 'Compositional'], default='Transformer',
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--relation-type', type=str, default='binary',
                    help='what kind of relations to learn. options: binary, ternary (default: binary)')
parser.add_argument('--transformer-dim', type=int, default=128)
parser.add_argument('--qk-dim', type=int, default=16)
parser.add_argument('--iterations', default=1, type=int,
                    help='Number of Transformer Iterations to use in the relational base')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--n-rules', type=int, default=1)
parser.add_argument('--dot', action='store_true', default=False)
parser.add_argument('--name', type=str, default='Default')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

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

wandb.init(settings=wandb.Settings(start_method='fork'),
           project="Sort-of-CLEVR-Gumbel", config=config,
           name=args.name)

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

folder_name = f'logs/{args.name}'
tensorboard_dir = f'{folder_name}/tensorboard/'
model_dir = f'{folder_name}/checkpoints/'

if not os.path.isdir(folder_name):
    os.makedirs(folder_name)
    os.makedirs(tensorboard_dir)
    os.makedirs(model_dir)

summary_writer = SummaryWriter(tensorboard_dir)

model = Model(args)

print(model)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of Parameters: ", n_params)

wandb.watch(model)

bs = args.batch_size
input_img = torch.FloatTensor(bs, 3, 75, 75)
input_qst = torch.FloatTensor(bs, 18)
label = torch.LongTensor(bs)

if args.cuda:
    model.cuda()
    input_img = input_img.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()

input_img = Variable(input_img)
input_qst = Variable(input_qst)
label = Variable(label)

def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

    
def train(epoch, ternary, rel, norel):
    model.train()

    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    random.shuffle(ternary)
    random.shuffle(rel)
    random.shuffle(norel)

    ternary = cvt_data_axis(ternary)
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    acc_ternary = []
    acc_rels = []
    acc_norels = []

    l_ternary = []
    l_binary = []
    l_unary = []

    last = len(rel[0]) // bs

    for batch_idx in range(last):
        tensor_data(ternary, batch_idx)
        accuracy_ternary, loss_ternary = model.train_(input_img, input_qst, label)
        acc_ternary.append(accuracy_ternary.item())
        l_ternary.append(loss_ternary.item())

        tensor_data(rel, batch_idx)
        accuracy_rel, loss_binary = model.train_(input_img, input_qst, label)
        acc_rels.append(accuracy_rel.item())
        l_binary.append(loss_binary.item())

        tensor_data(norel, batch_idx)
        accuracy_norel, loss_unary = model.train_(input_img, input_qst, label)
        acc_norels.append(accuracy_norel.item())
        l_unary.append(loss_unary.item())

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] '
                  'Ternary accuracy: {:.0f}% | Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(
                   epoch,
                   batch_idx * bs * 2,
                   len(rel[0]) * 2,
                   100. * batch_idx * bs / len(rel[0]),
                   accuracy_ternary,
                   accuracy_rel,
                   accuracy_norel))

    avg_acc_ternary = sum(acc_ternary) / len(acc_ternary)
    avg_acc_binary = sum(acc_rels) / len(acc_rels)
    avg_acc_unary = sum(acc_norels) / len(acc_norels)

    summary_writer.add_scalars('Accuracy/train', {
        'ternary': avg_acc_ternary,
        'binary': avg_acc_binary,
        'unary': avg_acc_unary
    }, epoch)

    avg_loss_ternary = sum(l_ternary) / len(l_ternary)
    avg_loss_binary = sum(l_binary) / len(l_binary)
    avg_loss_unary = sum(l_unary) / len(l_unary)

    summary_writer.add_scalars('Loss/train', {
        'ternary': avg_loss_ternary,
        'binary': avg_loss_binary,
        'unary': avg_loss_unary
    }, epoch)

    # return average accuracy
    return avg_acc_ternary, avg_acc_binary, avg_acc_unary

def test(epoch, ternary, rel, norel, split='Test'):
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    ternary = cvt_data_axis(ternary)
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    accuracy_ternary = []
    accuracy_rels = []
    accuracy_norels = []

    loss_ternary = []
    loss_binary = []
    loss_unary = []

    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(ternary, batch_idx)
        acc_ter, l_ter = model.test_(input_img, input_qst, label)
        accuracy_ternary.append(acc_ter.item())
        loss_ternary.append(l_ter.item())

        tensor_data(rel, batch_idx)
        acc_bin, l_bin = model.test_(input_img, input_qst, label)
        accuracy_rels.append(acc_bin.item())
        loss_binary.append(l_bin.item())

        tensor_data(norel, batch_idx)
        acc_un, l_un = model.test_(input_img, input_qst, label)
        accuracy_norels.append(acc_un.item())
        loss_unary.append(l_un.item())

    accuracy_ternary = sum(accuracy_ternary) / len(accuracy_ternary)
    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    print('{} set: Ternary accuracy: {:.0f}% Binary accuracy: {:.0f}% | Unary accuracy: {:.0f}%'.format(
        split, accuracy_ternary, accuracy_rel, accuracy_norel))

    summary_writer.add_scalars(f'Accuracy/{split}', {
        'ternary': accuracy_ternary,
        'binary': accuracy_rel,
        'unary': accuracy_norel
    }, epoch)

    loss_ternary = sum(loss_ternary) / len(loss_ternary)
    loss_binary = sum(loss_binary) / len(loss_binary)
    loss_unary = sum(loss_unary) / len(loss_unary)

    summary_writer.add_scalars('Loss/test', {
        'ternary': loss_ternary,
        'binary': loss_binary,
        'unary': loss_unary
    }, epoch)

    return accuracy_ternary, accuracy_rel, accuracy_norel

    
def load_data():
    print('loading data...')
    dirs = '/miniscratch/mittalsa/data/data'
    # dirs = '/miniscratch/mittalsa/data/old_data'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    with open(filename, 'rb') as f:
      train_datasets, val_datasets, test_datasets = pickle.load(f)

    ternary_train = []
    ternary_val = []
    ternary_test = []

    rel_train = []
    rel_val = []
    rel_test = []

    norel_train = []
    norel_val = []
    norel_test = []

    print('processing data...')

    for img, ternary, relations, norelations in train_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst, ans in zip(ternary[0], ternary[1]):
            ternary_train.append((img,qst,ans))
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, ternary, relations, norelations in val_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst, ans in zip(ternary[0], ternary[1]):
            ternary_val.append((img, qst, ans))
        for qst,ans in zip(relations[0], relations[1]):
            rel_val.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_val.append((img,qst,ans))

    for img, ternary, relations, norelations in test_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst, ans in zip(ternary[0], ternary[1]):
            ternary_test.append((img, qst, ans))
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))

    return (ternary_train, ternary_val, ternary_test, rel_train, rel_val, rel_test, norel_train, norel_val, norel_test)

ternary_train, ternary_val, ternary_test, rel_train, rel_val, rel_test, norel_train, norel_val, norel_test = load_data()

best_val_ternary, best_val_rel, best_val_norel = float('-inf'), float('-inf'), float('-inf')
opt_ternary, opt_rel, opt_norel = float('-inf'), float('-inf'), float('-inf')

with open(f'./{folder_name}/log.csv', 'w') as log_file:
    csv_writer = csv.writer(log_file, delimiter=',')
    csv_writer.writerow(['epoch',
                         'train_acc_ternary', 'train_acc_rel', 'train_acc_norel',
                         'val_acc_ternary', 'val_acc_rel', 'val_acc_norel',
                         'best_val_acc_ternary', 'best_val_acc_rel', 'best_val_acc_norel',
                         'test_acc_ternary', 'test_acc_rel', 'test_acc_norel',
                         'optimal_test_acc_ternary', 'optimal_test_acc_rel', 'optimal_test_acc_norel'])

    print(f"Training {args.model} model...")
    for epoch in range(1, args.epochs + 1):

        train_acc_ternary, train_acc_binary, train_acc_unary = train(
            epoch, ternary_train, rel_train, norel_train)
        print()

        val_acc_ternary, val_acc_binary, val_acc_unary = test(
            epoch, ternary_val, rel_val, norel_val, split='Val')
        test_acc_ternary, test_acc_binary, test_acc_unary = test(
            epoch, ternary_test, rel_test, norel_test, split='Test')

        if val_acc_ternary > best_val_ternary:
            best_val_ternary = val_acc_ternary
            opt_ternary = test_acc_ternary

        if val_acc_binary > best_val_rel:
            best_val_rel = val_acc_binary
            opt_rel = test_acc_binary

        if val_acc_unary > best_val_norel:
            best_val_norel = val_acc_unary
            opt_norel = test_acc_unary

        dict = {"Ternary Train Accuracy": train_acc_ternary,
                "Binary Train Accuracy": train_acc_binary,
                "Unary Train Accuracy": train_acc_unary,
                "Ternary Val Accuracy": val_acc_ternary,
                "Binary Val Accuracy": val_acc_binary,
                "Unary Val Accuracy": val_acc_unary,
                "Best Ternary Val Accuracy": best_val_ternary,
                "Best Binary Val Accuracy": best_val_rel,
                "Best Unary Val Accuracy": best_val_norel,
                "Ternary Test Accuracy": test_acc_ternary,
                "Binary Test Accuracy": test_acc_binary,
                "Unary Test Accuracy": test_acc_unary,
                "Optimal Ternary Test Accuracy": opt_ternary,
                "Optimal Binary Test Accuracy": opt_rel,
                "Optimal Unary Test Accuracy": opt_norel,
                }
        print()
        wandb.log(dict, step=epoch)

        csv_writer.writerow([epoch,
                             train_acc_ternary, train_acc_binary, train_acc_unary,
                             val_acc_ternary, val_acc_binary, val_acc_unary,
                             best_val_ternary, best_val_rel, best_val_norel,
                             test_acc_ternary, test_acc_binary, test_acc_unary,
                             opt_ternary, opt_rel, opt_norel])

        model.save_model(epoch, model_dir)