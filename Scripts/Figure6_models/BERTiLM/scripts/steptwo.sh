#!/bin/bash

echo $CONDA_DEFAULT_ENV

python3 ../../swing_roberta/train_tokenizer.py \
        --data_dir encoded_txt \
        --out_dir tokenizer \
        --vocab_size 16 \
        --max_size 1 \
        --tokenizer by-digit
