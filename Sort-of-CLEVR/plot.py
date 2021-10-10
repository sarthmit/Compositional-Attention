from __future__ import print_function
import argparse
import os
import pandas as pd
import glob

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.font_manager

sns.color_palette("flare", as_cmap=True)
sns.set(style="whitegrid")

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

df = pd.DataFrame(columns=["Model", "Iterations", "Heads", "Rules", \
                           "Dimension", "qk-dim", "Dot", "Seed", \
                           "Question Type", "Test Accuracy"])

files = glob.glob(f'{args.path}/*')
for file in files:
    name = file.split('/')[-1].split('_')

    if "dot" in file:
        dot = True
    else:
        dot = False

    model = name[1]
    iter = int(name[2])
    dim = int(name[3])
    searches = int(name[4])
    retrievals = int(name[5])
    qk_dim = int(name[6])
    seed = int(name[7])

    if dim == 512:
        continue
    if model == 'Compositional':
        if dim == 32 and dot == True and retrievals == 2 and qk_dim == 8:
            pass
        elif dim == 256 and dot == True and retrievals == 2 and qk_dim == 32:
            pass
        elif dim == 512 and dot == True and retrievals == 2 and qk_dim == 16:
            pass
        else:
            continue

    try:
        with open(f'{file}/log.csv', 'r') as f:
            data = f.read().split('\n')[:-1][-1].split(',')

        unary = float(data[-1])
        binary = float(data[-2])
        ternary = float(data[-3])

        df.loc[-1] = [model, iter, searches, retrievals, dim, qk_dim, dot, seed, \
                      "Unary", unary]
        df.index = df.index + 1
        df.loc[-1] = [model, iter, searches, retrievals, dim, qk_dim, dot, seed, \
                      "Binary", binary]
        df.index = df.index + 1
        df.loc[-1] = [model, iter, searches, retrievals, dim, qk_dim, dot, seed, \
                      "Ternary", ternary]
        df.index = df.index + 1
    except:
        print(file)

print(df)
g = sns.catplot(x="Question Type", y="Test Accuracy", col="Dimension", \
                hue = "Model", data=df, \
                kind="bar", palette="mako", \
                alpha=0.8, saturation=0.6)

(g.set_titles("Model Dimension {col_name}")
 .set(ylim=(40, 100))
 .despine(left=True))

# plt.tight_layout()
plt.savefig("test.png", bbox_inches='tight')
plt.close()
