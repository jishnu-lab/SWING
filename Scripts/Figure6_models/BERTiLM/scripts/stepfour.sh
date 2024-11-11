#!/bin/bash

echo $CONDA_DEFAULT_ENV

python3 ../../swing_roberta/pretrain.py \
      --train pretrain/pretrain_train_trunc4500 \
      --eval pretrain/pretrain_eval_trunc4500 \
      --run_name sep20_start \
      --max_size 4500 \
      --vocab_size 16 \
      --project_name npflip_pretrain_bydigit \
      --save_path pretrain/checkpoints \
      --epochs 40 \
      --tokenizer tokenizer/tokenizer_vocab16_max7619.json \
      --per_device 4 \
      --hidden_size 128 \
      --hidden_depth 4 \
      --attention_heads 4 \
      --intermed_size 64 \
      --lr 0.0001

