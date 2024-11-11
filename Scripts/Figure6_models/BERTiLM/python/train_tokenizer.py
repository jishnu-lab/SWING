"""
Build and train a byte-pair encoding tokenizer. Saves to out_dir as .json file.
REQUIRES: data_dir   - a string; path to directory of training data files
          out_dir    - a string; path to directory in which to save output files
          vocab_size - an integer; the number of unique tokens in vocabulary (its cardinality)
          max_size   - an integer; the index of the maximum length sequence
          tokenizer  - a string; tokenizer type (either 'bpe' or 'by-digit')
MODIFIES: none
RETURNS:  none
"""

import argparse
import csv    
import glob
import os
import pathlib
import re

import pandas as pd

from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    ByteLevelBPETokenizer
)


def main():
    parser = argparse.ArgumentParser(prog = 'Train BPE tokenizer.',
                                     description = 'Creates and trains a BPE tokenizer.')
    parser.add_argument('-d', '--data_dir', type = str, help = 'data directory')
    parser.add_argument('-o', '--out_dir', type = str, help = 'output directory')
    parser.add_argument('-v', '--vocab_size', type = int, help = 'number of unique tokens in vocabulary')
    parser.add_argument('-m', '--max_size', type = int, default = 1, help = 'index of maximum length sequence (1 = longest sequence, while 2 = second longest)')
    parser.add_argument('-t', '--tokenizer', type = str, default = 'bpe', help = 'tokenizer type')
    args = parser.parse_args()
    
    ## Get available training data in .txt files in provided data directory
    path = args.data_dir
    paths = [str(x) for x in pathlib.Path(path).glob('*.txt')]

    if args.tokenizer == 'bpe':
        ## Initialize a BPE tokenizer
        tokenizer = Tokenizer(models.BPE())

        ## Set up pre-tokenizer step and BPE trainer
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size     = args.vocab_size, 
                                      special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<|endoftext|>"], 
                                      show_progress  = True)
    else:
        ## Initialize a WordLevel tokenizer with by-digit splitting
        ## Vocabulary is just the digits 0-9
        tokenizer = Tokenizer(models.WordLevel())

        ## Set up pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.Digits(individual_digits = True)
        trainer = trainers.WordLevelTrainer(vocab_size = args.vocab_size,
                                            special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<|endoftext|>"],
                                            show_progress = True)
    
    
    ## Load training data as a dataset
    dataset = load_dataset("text", data_files = paths)
    
    ## Train the tokenizer
    tokenizer.train(files = paths, trainer = trainer)
    
    ## Add a post-processor that prepends a start token and appends an end token
    tokenizer.post_processor = processors.RobertaProcessing(sep = ('</s>', 2), cls = ('<s>', 0))
    
    ## Tokenize all training data and find longest sequence
    encoded_train = [len(tokenizer.encode(sample['text'])) for sample in dataset['train']]
    encoded_train.sort()
    longest = encoded_train[-args.max_size]
    print('Longest: ')
    print(encoded_train[-49:])

    ## Save tokenizer 
    tokenizer_file = 'tokenizer_vocab' + str(args.vocab_size) + '_max' + str(longest) + '.json'
    out_path = os.path.join(args.out_dir, tokenizer_file)
    
    ## Create output directory if it doesn't exist already
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    tokenizer.save(out_path)
    
    
if __name__ == "__main__":
    main()
