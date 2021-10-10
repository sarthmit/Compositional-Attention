#!/bin/bash

model=$1
seq_len=$2
dim=$3
search_dim=$4
value_dim=$5
search=$6
v_s=$7
retrieve=$8
v_p=$9
seed=${10}
extras=${11}

python main.py --seq-len $seq_len --model $model --dim $dim \
--search-dim $search_dim --value-dim $value_dim --search $search \
--retrieve $retrieve --v-s $v_s --v-p $v_p --seed $seed $extras

# run.sh Compositional-dot 10 64 64 64 2 2 4 4 1 --concat