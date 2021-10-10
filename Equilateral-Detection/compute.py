from __future__ import print_function
import argparse
import os
import pandas as pd
import glob

pd.set_option("display.max_rows", None, "display.max_columns", None)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

df = pd.DataFrame(columns=["Model", "Dataset", "Iterations", "Heads", "Rules", \
                           "Dimension", "qk-dim", "Dot", "Learning Rate", "Seed", \
                           "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"])

files = glob.glob(f'{args.path}/*')
for file in files:
    name = file.split('/')[-1].split('_')

    if "dot" in file:
        dot = True
    else:
        dot = False

    model = name[2]
    dataset = name[1]
    iter = name[3]
    dim = name[4]
    searches = name[5]
    retrievals = name[6]
    qk_dim = name[7]
    lr = 0.0001
    seed = name[8]

    try:
        with open(f'{file}/log.csv', 'r') as f:
            data = f.read().split('\n')[:-1][-1].split(',')

        if '200' not in data[0]:
            print(file)
            continue

        trl = float(data[-4])
        tra = float(data[-3])
        tel = float(data[-2])
        tea = float(data[-1])

        df.loc[-1] = [model, dataset, iter, searches, retrievals, dim, qk_dim, dot, lr, seed, \
                      trl, tra, tel, tea]
        df.index = df.index + 1
    except:
        print(file)

print(df)
print(df.groupby(["Dataset", "Model", "Iterations", "Dimension", "Heads", \
                  "qk-dim", "Rules", "Dot", "Learning Rate"])['Test Accuracy'].mean())
# print(df.groupby(["Dataset", "Model", "Iterations", "Dimension", "Heads", \
#                   "qk-dim", "Rules", "Dot"])['Test Accuracy'].std())
print(df.groupby(["Dataset", "Model", "Iterations", "Dimension", "Heads", \
                  "qk-dim", "Rules", "Dot", "Learning Rate"])['Test Accuracy'].count())