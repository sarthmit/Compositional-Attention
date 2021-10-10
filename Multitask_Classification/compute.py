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
                           "Dimension", "qk-dim", "Dot", "Seed", \
                           "Params", "Loss", "Accuracy"])

files = glob.glob(f'{args.path}/*')
for file in files:
    name = file.split('/')[-1].split('_')

    if "dot" in file:
        dot = True
    else:
        dot = False

    model = name[1]
    rules = name[2]
    heads = name[3]
    dim = name[4]
    qk = name[5]
    iter = name[6]
    seed = name[7]

    with open(f'{file}/log.txt', 'r') as f:
        data = f.read().split('\n')
        np = data[0]
        data = data[-11:-1]

    np = float(np.split(':')[1].split('E')[0])

    if '100' not in data[0]:
        print(file)
        continue

    for line in data:
        if 'Epoch' in line:
            continue
        seg = line.split(' | ')
        if 'Train' in line:
            continue
        df.loc[-1] = [model, seg[1], iter, heads, rules, dim, qk, dot, seed, np, float(seg[-1].split(' ')[-1]), float(seg[-2].split(' ')[-1])]
        df.index = df.index + 1

grouped_df = df.groupby(["Iterations", "Dimension", "Heads", "Model", \
                         "Rules", "qk-dim", "Dot", "Dataset"])
#print(grouped_df[["Params", "Accuracy", "Loss"]].mean())
#print(grouped_df[["Params", "Accuracy", "Loss"]].std())
print(grouped_df[["Accuracy"]].mean())
print(grouped_df[["Accuracy"]].std())
print(grouped_df['Loss'].count())
