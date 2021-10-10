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

name="VIT_"$model"_"$rules"_"$heads"_"$dim"_"$qk_dim"_"$iterations"_"$seed""$ext

echo Running on $HOSTNAME
echo Running version $name
echo Extra arguments: $extras

python main_multi.py --model $model --n-heads $heads --n-rules $rules \
--transformer-dim $dim --seed $seed --qk-dim $qk_dim --iterations $iterations \
--name $name $extras
