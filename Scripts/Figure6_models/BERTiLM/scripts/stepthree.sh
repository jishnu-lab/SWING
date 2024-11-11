#!/bin/bash

echo $CONDA_DEFAULT_ENV

python3 ../../swing_roberta/build_pretrain_data.py \
    --train_dir encoded_txt \
    --tokenizer tokenizer/tokenizer_vocab16_max7619.json \
    --out_dir pretrain \
    --max_size 4500 \
    --split_prop 0.15
    
      
