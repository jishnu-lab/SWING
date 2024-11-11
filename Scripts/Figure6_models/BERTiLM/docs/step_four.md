# Step Four: pretraining RoBERTa 

In contrast to BERT, pretraining RoBERTa uses only a masked language modeling objective. To put it simply, tokens in each training example are randomly chosen to be covered up. The model is then asked to predict what token was masked. There are a few other differences, but the reader can turn to the original RoBERTa paper for details. To perform the pretraining steps, we use pretrain.py.

***pretrain.py***:
To pretrain on data created with build_pretrain_data.py, run the following command and change the parts in brackets as desired:
```
python3 pretrain.py --train [path/to/training/set/arrow/files] \
                    --eval [path/to/evaluation/set/arrow/files] \
                    --tokenizer [path/to/pretrained/tokenizer] \
                    --run_name [name of run] \
                    --save_path [path/where/you/want/to/save/the/model/checkpoints] \
                    --max_size [index of longest sequence length to use for truncation] \
                    --epochs [number of epochs] \
                    --per_device [number of samples per device]
```

There are many other arguments that can be provided in the function call to change things like the model architecture and the ways the training is logged. The default settings have been sufficient for our purposes up until the writing of this documentation, but we invite the reader to look at the docstring of pretrain.py for details on what other customizable options there are. 

It's also important to note that this is the most computation-heavy step, so make sure you have the resources and time. 

## Example:
```    
python3 swing_roberta/pretrain.py \
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

```