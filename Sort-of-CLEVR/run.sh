#!/bin/bash

model=$1
iterations=$2
dim=$3
heads=$4
qk_dim=$5
rules=$6
seed=$7
extras=$8

ext=${extras//\-\-/\_}
ext=${ext// /}

name="VIT_"$model"_"$iterations"_"$dim"_"$heads"_"$rules"_"$qk_dim"_"$seed""$ext

echo Running on $HOSTNAME
echo Running version $name
echo Extra arguments: $extras

PYTHONUNBUFFERED=1 python main.py --epochs 100 --relation-type binary --model $model \
--transformer-dim $dim --n-heads $heads --n-rules $rules --seed $seed \
--iterations $iterations --name $name --qk-dim $qk_dim $extras

# run.sh Compositional 4 256 4 32 1 $seed '--dot'