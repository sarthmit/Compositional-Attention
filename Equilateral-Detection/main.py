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

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Image Classification Transformer')
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

parser.add_argument('--transformer-dim', type=int, default=128)
parser.add_argument('--qk-dim', type=int, default=16)
parser.add_argument('--iterations', default=1, type=int,
                    help='Number of Transformer Iterations to use in the relational base')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--n-rules', type=int, default=1)
parser.add_argument('--dot', action='store_true', default=False)

parser.add_argument('--name', type=str, default='Default')
args = parser.parse_args()

args.dataset = 'Triangles'
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
    "data": args.dataset,
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

train_dataset = TriangleDataset(num_examples = 50000)
test_dataset = TriangleDataset(num_examples = 10000)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, num_workers = 2, shuffle = False)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, num_workers = 2, shuffle = False)

device = 'cuda' if args.cuda else 'cpu'

net = Model(args)
net = net.to(device)

print(net)
n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Number of Parameters: ", n_params)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train():
    net.train()

    correct = 0.
    total_loss = 0.
    total = 0.

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        outputs = outputs.squeeze()
        prediction = (torch.sigmoid(outputs) >= 0.5).int()
        targets = targets.float()

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        correct += torch.eq(prediction, targets).sum().item()

    loss = total_loss / total
    accuracy = correct / total

    return loss, accuracy * 100

def test():
    net.eval()

    correct = 0.
    total_loss = 0.
    total = 0.

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs = outputs.squeeze()
            prediction = (torch.sigmoid(outputs) >= 0.5).int()
            targets = targets.float()

            loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            correct += torch.eq(prediction, targets).sum().item()

    loss = total_loss / total
    accuracy = correct / total

    return loss, accuracy * 100

test_loss, test_acc = test()
print(f"Starting Loss: {test_loss:.3f}  |  Starting Accuracy: {test_acc:.2f}")
print()

with open(f'./{folder_name}/log.csv', 'w') as log_file:
    csv_writer = csv.writer(log_file, delimiter=',')
    csv_writer.writerow(['Epoch',
                         'Train Loss', 'Train Accuracy', \
                         'Test Loss', 'Test Accuracy'])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train()
        test_loss, test_acc = test()

        dict = {"Train Loss": train_loss,
                "Train Accuracy": train_acc,
                "Test Loss": test_loss,
                "Test Accuracy": test_acc}

        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.3f}   |   Train Accuracy: {train_acc:.2f}")
        print(f"Test Loss: {test_loss:.3f}   |   Test Accuracy: {test_acc:.2f}")
        print()

        scheduler.step()

        csv_writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
        torch.save(net.state_dict(), os.path.join(model_dir, 'model.pt'))