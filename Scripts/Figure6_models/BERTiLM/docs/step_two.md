# Step Two: training a tokenizer 

In the case of SWING RoBERTa, we use a byte-pair encoding (BPE) tokenizer, which must be trained. During training, all of the samples are broken down into single character pieces and then iteratively combined into pairs based upon their frequency. Usually, this training is quite fast, so if it is taking a long time, something may be wrong with your dataset. The training is done with train_tokenizer.py, which trained a BPE tokenizer and saves it to some specified directory.

***train_tokenizer.py***:
To train a new BPE tokenizer, run the following command and change the parts in brackets as desired:
```
python3 train_tokenizer.py --data_dir [path/to/data/to/train/on] \
                           --out_dir [path/where/you/want/to/save/the/file] \
                           --vocab_size [number of unique tokens in vocabulary] \
                           --max_size [index of longest sequence length to use for truncation] \
                           --tokenizer [by-digit or bpe]
```
        
The second to last argument, `max_size`, is the index of the longest sequence length to use for truncation. In the original BERT paper, they truncated all sequences to a length of 512 tokens. Since we have to consider biological context, we do not want to truncate too much. If `max_size = 1`, this will set the maximum sequence length to the number of tokens in the longest encoded example, which is effectively no truncation. If `max_size = 2`, this will set the maximum sequence length to the number of tokens in the second longest encoded example.
 
## Example
```
python3 swing_roberta/train_tokenizer.py \
        --data_dir encoded_txt \
        --out_dir tokenizer \
        --vocab_size 16 \
        --max_size 1 \
        --tokenizer by-digit
```

This trains the by-digit tokenizer on the data that was cleaned and formatted in step one. The vocabulary size is set to 16 tokens (since we only have 16 different ones --- special tokens + digits 0 through 9) and the maximum number of tokens in one sequence is set to the maximum encoded length of any sequence trained upon. 


The default for RoBERTa was 50,265 vocabulary size.