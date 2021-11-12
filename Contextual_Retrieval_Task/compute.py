import pandas as pd

pd.set_option('display.max_rows', None)

import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Toy Task for Transformer')
parser.add_argument('--v-s', type=int, default=1)
parser.add_argument('--v-p', type=int, default=2)
parser.add_argument('--seq-len', type=int, default=10)
parser.add_argument('--no-coeff', action='store_true', default=False)
args = parser.parse_args()

df = pd.DataFrame(columns=["Sequence Length", "MLP Type", "Type", "Number of Parameters", "Search", "Retrieval", \
                           "Dimension", "Search Dimension", "Value Dimension", "Concat", "Gumbel", "Separate", "Seed", "Loss"])

if args.no_coeff:
    folders = ["No_Coeff_Trained_Models"]
else:
    folders = ["Trained_Models"]

for folder in folders:
    tasks = os.listdir(folder)
    for task in tasks:
        task_split = task.split('_')
        seq_len = int(task_split[0])
        vs = int(task_split[1])
        vp = int(task_split[2])

        if seq_len != args.seq_len or args.v_s != vs or args.v_p != vp:
            continue

        seeds = os.listdir(f'{folder}/{task}')

        for seed in seeds:
            trials = os.listdir(f'{folder}/{task}/{seed}')

            for trial in trials:
                name = f'{folder}/{task}/{seed}/{trial}'
                if not os.path.exists(f'{name}/log.txt'):
                    print(name)
                    continue

                with open(f'{name}/log.txt') as f:
                    data = f.read().split('\n')[:-1]
                    params = int(data[0].split(':')[-1])
                    if '100000' not in data[-1]:
                        continue
                    perf = float(data[-1].split('|')[1].split(':')[-1])

                name_pieces = name.split('/')[-1].split('_')

                dim = float(name_pieces[1])
                search_dim = float(name_pieces[2])
                value_dim = float(name_pieces[3])
                concat = 'concat' in name_pieces
                gumbel = 'gumbel' in name_pieces
                mlp = folder.split('_')[-1]
                type = name_pieces[0]
                search = float(name_pieces[4])
                retrieval = float(name_pieces[5])

                if type == "Compositional-dot":
                    if "separate" not in name_pieces:
                        separate = False
                    else:
                        separate = True
                else:
                    separate = False

                entry = [float(seq_len), mlp, type, params, search, retrieval, dim, search_dim, value_dim, concat, gumbel, separate, seed, perf]
                df.loc[-1] = entry
                df.index = df.index + 1

print(df)
print()
print(df.groupby(["Sequence Length", "MLP Type", "Type", "Number of Parameters", "Search", "Retrieval", \
                           "Dimension", "Search Dimension", "Value Dimension", "Concat", "Gumbel", "Separate"])['Loss'].mean())
print(df.groupby(["Sequence Length", "MLP Type", "Type", "Number of Parameters", "Search", "Retrieval", \
                           "Dimension", "Search Dimension", "Value Dimension", "Concat", "Gumbel", "Separate"])['Loss'].count())
