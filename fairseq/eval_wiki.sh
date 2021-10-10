#!/bin/bash

path=$1

fairseq-eval-lm data-bin/wikitext-103 \
    --path $path \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400