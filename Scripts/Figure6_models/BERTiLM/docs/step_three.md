# Step Three: creating your pretraining datasets

To facilitate pretraining, we convert our data files, which are usually .txt or .csv format, into Hugging Face datasets. This makes it possible to use built-in functions for training instead of having to code it all up ourselves. We create our pretraining datasets with build_pretrain_data.py.


***build_pretrain_data.py***:
To create pretraining datasets from existing files, run the following command and change the parts in brackets as desired:
```
python3 build_pretrain_data.py --train_dir [path/to/training/data] \
                               --tokenizer [path/to/pretrained/tokenizer] \
                               --out_dir [path/where/you/want/to/save/the/files] \
                               --max_size [index of longest sequence length to use for truncation] \
                               --split_prop [proportion of training data to use for evaluation]
```

This will read in the data provided in `train_dir` (and, optionally, `eval_dir`) and saves the created Hugging Face datasets (.arrow format) in the specified `out_dir` directory.

## Example:
```
python3 swing_roberta/build_pretrain_data.py \
    --train_dir encoded_txt \
    --tokenizer tokenizer/tokenizer_vocab16_max7619.json \
    --out_dir pretrain \
    --max_size 4500 \
    --split_prop 0.15
```
