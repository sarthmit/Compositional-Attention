#!/bin/bash

type=$1
mode=$2
layers=$3
embdim=$4
ffndim=$5
attn_dim=$6
sdim=$7
heads=$8
rules=$9
lr=${10}
seed=${11}
extras=${12}

ext=${extras//\-\-/\_}
ext=${ext// /}

base=$mode"_"$layers"_"$embdim"_"$ffndim

if [[ $type == "Stacked" ]]; then
  base="Stacked_"$base
  arch="transformer"
elif [[ $type == "Universal" ]]; then
  base="Universal_"$base
  arch="universal_transformer"
fi

base="Wiki103/"$base

if [[ $mode == "Compositional" ]]; then
	base=$base"_"$attn_dim"_"$sdim"_"$heads"_"$rules
elif [[ $mode == "Compositional_func" ]]; then
	base=$base"_"$attn_dim"_"$sdim"_"$heads"_"$rules
elif [[ $mode == "Standard" ]]; then
	base=$base"_"$heads
fi

base=$base"_"$lr"_"$seed

base=$base""$ext
mkdir -p "logs/Wiki103/"

name="checkpoints/"$base
tb="tensorboard/"$base
base="logs/"$base
wandb="Wiki103"

echo Running name is $name

PYTHONUNBUFFERED=1 python3 fairseq_cli/train.py \
    --task language_modeling \
    data-bin/wikitext-103 \
    --save-dir checkpoints/transformer_wikitext-103 \
    --tensorboard-logdir $tb \
    --wandb-project $wandb \
    --arch universal_transformer_lm --share-decoder-input-output-embed \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
    --lr $lr --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --tokens-per-sample 512 --sample-break-mode none \
    --max-tokens 2048 --update-freq 16 \
    --fp16 \
    --max-update 50000 \
    --decoder-embed-dim	$embdim \
    --decoder-ffn-embed-dim $ffndim \
    --decoder-layers $layers \
    --decoder-attention-heads $heads \
    --attention-rules $rules \
    --attention-type $mode \
    --attn-dim $attn_dim \
    --selection-dim $sdim \
    $extras \
    --seed $seed \
    --save-dir $name > $base".log"

# sbatch --gres=gpu:rtx8000:1 --mem=32G run_wiki.sh Universal Standard 6 512 2048 512 -1 8 -1 0.0005
# sbatch --gres=gpu:rtx8000:1 --mem=32G run_wiki.sh Universal Compositional 6 512 2048 512 32 8 8 0.0005 '--qk-rule'
# sbatch --gres=gpu:rtx8000:1 --mem=32G run_wiki.sh Universal Compositional_func 6 512 2048 512 32 8 8 0.0005 '--nonlinear'