"""
Extracts embeddings for datasets.
REQUIRES: data           - a string; the path to data
          eval            - a string; the path to the evaluation dataset(s)
          tokenizer       - a string; the path to the pretrained tokenizer
          model           - a string; the path to a pretrained model
          save_path       - a string; the path to the directory in which to save the model checkpoints
RETURNS:  none
"""
import argparse
import evaluate
import math
import os
import pathlib
import sys
import torch

from transformers import RobertaModel, PreTrainedTokenizerFast
from tqdm import tqdm

import numpy as np
import pandas as pd



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type = str, help = 'path to dataset file(s)')
    parser.add_argument('--tokenizer', type = str, help = 'path to pretrained tokenizer')
    parser.add_argument('--model', type = str, help = 'path to pretrained model')
    parser.add_argument('--save_path', type = str, help = 'path to save to', dest = 'model_path')  
    parser.add_argument('--max_length', type = int, help = 'max sequence length', default = 4500)

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ## load pretrained tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = args.tokenizer,
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

    
    ## load model
    model = RobertaModel.from_pretrained(args.model, output_hidden_states = True)

    ## load datasets
    data_files = [str(x) for x in pathlib.Path(args.data).glob('*.txt')]
    data = []
    for file in data_files:
        with open(file) as f:
            for line in f:
                data.append(line)

    model = model.to(device)
    model.eval()

    def embed(example):
      tokenized = tokenizer.encode_plus(example, padding = 'max_length', truncation = True, max_length = args.max_length, add_special_tokens = False, return_tensors = 'pt')
      input_ids = tokenized['input_ids']
      with torch.no_grad():
        output = model(input_ids = input_ids)
      hidden_states = output[0]
      embedded = torch.mean(hidden_states, dim = 1).squeeze()
      return embedded.detach().numpy()

    embeddings = list(map(embed, tqdm(data)))
    embedding_array = pd.DataFrame(np.vstack(embeddings))
    embedding_array.to_csv('embedding.csv')


    #tokenized = tokenizer.encode_plus(data, add_special_tokens = False, return_tensors = 'pt')
   # inputs = tokenized['input_ids'] 
    #print(inputs)
    # ## reshape
    # tokenized = torch.LongTensor(tokenized)
    # tokenized = tokenized.to(device)
    # tokenized = tokenized.unsqueeze(0)
    # ## get embedded sequence
    # with torch.no_grad():
    #     output = model(input_ids = tokenized)
    # print(output)

if __name__ == "__main__":
    main()
