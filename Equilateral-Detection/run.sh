#!/bin/bash

model=$1
iterations=$3
dim=$4
heads=$5
qk_dim=$6
rules=$7
lr=$8
seed=$9
extras=${10}

ext=${extras//\-\-/\_}
ext=${ext// /}

name="VIT_"$model"_"$iterations"_"$dim"_"$heads"_"$rules"_"$qk_dim"_"$lr"_"$seed""$ext

echo Running on $HOSTNAME
echo Running version $name
echo Extra arguments: $extras

python main.py --epochs 200 --model $model --lr $lr \
--transformer-dim $dim --n-heads $heads --n-rules $rules --seed $seed \
--iterations $iterations --name $name --qk-dim $qk_dim $extras

# ./run.sh Compositional 4 256 4 32 2 0.0001 1 '--dot'