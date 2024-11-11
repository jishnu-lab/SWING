"""
Create Hugging Face datasets for pretraining.
Saves datasets as .arrow files in the specified output directory.
REQUIRES: train_dir  - a string; path to directory containing training data
          tokenizer  - a string; path to a pretrained tokenizer
          split_prop - a float; the proportion of the training data to use for evaluation if eval_dir is not provided
          max_size   - an integer; the maximum number of tokens in the tokenized sequences
          out_dir    - a string; path to directory in which to save output files
MODIFIES: none
RETURNS:  none
"""

import argparse
import csv    
import glob
import itertools
import os
import pathlib

import pandas as pd

from datasets import load_dataset
from padder import Padder
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

def main():
    parser = argparse.ArgumentParser(prog = 'Pretraining Dataset Builder.',
                                     description = 'Read in data and create dataset files for pretraining.')
    parser.add_argument('-d', '--train_dir', type = str, help = 'path to training data (.csv) files')
    parser.add_argument('-t', '--tokenizer', type = str, help = 'path to tokenizer (.json) file')
    parser.add_argument('-s', '--split_prop', type = float, help = 'proportion to use for evaluation if no directory specified', default = 0.2)
    parser.add_argument('-o', '--out_dir', type = str, help = 'path to directory in which to save datasets')
    parser.add_argument('-m', '--max_size', type = int, help = 'max length of sequences to truncate to')
    args = parser.parse_args()
    
    ## load in pretrained tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = args.tokenizer, 
                                        clean_up_tokenization_spaces = False,
                                        padding_side = 'right',
                                        truncation_side = 'right',
                                        pad_token = '<pad>',
                                        bos_token = '<s>',
                                        eos_token = '</s>',
                                        unk_token = '<unk>',
                                        cls_token = '<s>',
                                        sep_token = '</s>',
                                        mask_token = '<mask>',
                                        additional_special_tokens = ['<|endoftext|>'])
    
    ## load data
    train_files = [str(x) for x in pathlib.Path(args.train_dir).glob('*.txt')]
    dataset = load_dataset("text", data_files = {'train': train_files})
        
    ## tokenize data
    def tokenization(batched_text):
        return tokenizer(batched_text['text'], padding = 'max_length', max_length = args.max_size, truncation = True)
    
    train_data = dataset['train'].map(tokenization, batched = True, batch_size = 16) ##len(dataset['train']))

    ## format dataset
    train_dataset = train_data.remove_columns('text')
    train_dataset.set_format("torch", columns = ["input_ids", "attention_mask", "token_type_ids"])
    
    ## split into training and evaluation subsets
    dataset_dict = train_dataset.train_test_split(args.split_prop)
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict['test']

    ## save datasets
    train_file_name = 'pretrain_train_trunc' + str(args.max_size)
    eval_file_name = 'pretrain_eval_trunc' + str(args.max_size)
    train_dataset.save_to_disk(os.path.join(args.out_dir, train_file_name))
    eval_dataset.save_to_disk(os.path.join(args.out_dir, eval_file_name))
        

if __name__ == "__main__":
    main()
