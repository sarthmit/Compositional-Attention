from __future__ import print_function
import argparse
import os
import pandas as pd
import glob

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

df = pd.DataFrame(columns=["Model", "Iterations", "Heads", "Rules", \
                           "Dimension", "qk-dim", "Dot", "Seed", \
                           "Ternary Test", "Binary Test", "Unary Test"])

files = glob.glob(f'{args.path}/*')
for file in files:
    name = file.split('/')[-1].split('_')

    if "dot" in file:
        dot = True
    else:
        dot = False

    model = name[1]
    iter = name[2]
    dim = name[3]
    searches = name[4]
    retrievals = name[5]
    qk_dim = name[6]
    seed = name[7]

    try:
        with open(f'{file}/log.csv', 'r') as f:
            data = f.read().split('\n')[:-1][-1].split(',')

        if '100' not in data[0]:
            print(file)
            continue

        unary = float(data[-1])
        binary = float(data[-2])
        ternary = float(data[-3])

        df.loc[-1] = [model, iter, searches, retrievals, dim, qk_dim, dot, seed, \
                      ternary, binary, unary]
        df.index = df.index + 1
    except:
        print(file)

print(df)
print(df.groupby(["Model", "Iterations", "Dimension", "Heads", \
                  "qk-dim", "Rules", "Dot"])[['Unary Test', 'Binary Test', 'Ternary Test']].median())
print(df.groupby(["Model", "Iterations", "Dimension", "Heads", \
                  "qk-dim", "Rules", "Dot"])[['Unary Test', 'Binary Test', 'Ternary Test']].mean())
print(df.groupby(["Model", "Iterations", "Dimension", "Heads", \
                  "qk-dim", "Rules", "Dot"])[['Unary Test', 'Binary Test', 'Ternary Test']].std())
print(df.groupby(["Model", "Iterations", "Dimension", "Heads", \
                  "qk-dim", "Rules", "Dot"])['Unary Test'].count())