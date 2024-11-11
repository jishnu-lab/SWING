"""
Create vocabulary files from cleaned datasets.
Extracts window encodings and k-mers using specified arguments for step size and k.
REQUIRES: data_dir  - a string; path to directory containing datasets
          out_dir   - a string; path to directory in which to save output files
          k         - an integer; number of characters in a k-mer
          sub_size  - an integer; number of characters to skip when creating k-mers (skip_size number of characters will not overlap between two adjacent k-mers)
          l         - an integer; number of positions on either side of mutation position for sliding window
          type      - a string; the type of dataset (one of 'HLA' or 'MUTINT')
          freq      - a float; proportion of k-mers to keep by frequency or integer indicating frequency needed to not be filtered out
MODIFIES: none
RETURNS:  none
"""

import argparse
import csv    
import glob
import itertools
import os
import pathlib
import sys

import pandas as pd
import numpy as np

from collections import Counter
from itertools import repeat
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(prog = 'Vocabulary File Builder.',
                                     description = 'Read datasets and create vocabulary files.')
    parser.add_argument('-d', '--data_dir', help = 'path to directory containing datasets')
    parser.add_argument('-o', '--out_dir', help = 'path to directory in which to save output files')
    parser.add_argument('-k', '--k', type = int, default = 5, help = 'number of characters in a k-mer')
    parser.add_argument('-s', '--sub_size', type = int, default = 1, help = 'number of characters to skip when creating k-mers')
    parser.add_argument('-l', '--l', type = int, default = 2, help = 'number of positions on either side of mutation position for sliding window')
    parser.add_argument('-t', '--type', type = str, default = 'HLA', dest = 'type', help = 'the type of dataset')
    parser.add_argument('-f', '--freq', type = float, default = 0.8, dest = 'freq', help = 'what of top k-mers by frequency should be kept or integer indicating frequency needed to not be filtered out')
    args = parser.parse_args()
    
    ## create directory to save output if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(os.path.join(args.out_dir, 'kmers_csv')):    
    	os.makedirs(os.path.join(args.out_dir, 'kmers_csv'))
    if not os.path.exists(os.path.join(args.out_dir, 'kmers_csv', 'filt')):
        os.makedirs(os.path.join(args.out_dir, 'kmers_csv', 'filt'))
    if not os.path.exists(os.path.join(args.out_dir, 'window_encodings')):
    	os.makedirs(os.path.join(args.out_dir, 'window_encodings'))
    if not os.path.exists(os.path.join(args.out_dir, 'kmers_txt')):
        os.makedirs(os.path.join(args.out_dir, 'kmers_txt'))
    if not os.path.exists(os.path.join(args.out_dir, 'encoded_txt')):
        os.makedirs(os.path.join(args.out_dir, 'encoded_txt'))
    if not os.path.exists(os.path.join(args.out_dir, 'kmers_txt', 'filt')):
        os.makedirs(os.path.join(args.out_dir, 'kmers_txt', 'filt'))
    
    ## get list of data files
    if args.data_dir.endswith('.csv'):
        data_files = [args.data_dir]
    else:
        data_files = list(pathlib.Path(args.data_dir).glob('*.csv'))
    
    ## build amino acid grantham score dictionary
    grantham_scores_dict = {'A': 8.1,  'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5,
                            'E': 12.3, 'Q': 10.5, 'G': 9.0,  'H': 10.4, 'I': 5.2,
                            'L': 4.9,  'K': 11.3, 'M': 5.7,  'F': 5.2,  'P': 8.0,
                            'S': 9.2,  'T': 8.6,  'W': 5.4,  'Y': 6.2,  'V': 5.9}
    aa_score_dict = build_aa_score_dict(grantham_scores_dict)

    ## iterate through data files
    for file in tqdm(data_files, desc = 'Reading file', position = 1, leave = True):
        data_name = str(file).rsplit('/', 1)[-1].rsplit('.csv', 1)[0]
        print(data_name)
        window_file = '{0}_vocab_window_encodings.csv'.format(data_name)
        ## check if window encodings have already been created --- this is the most intensive step
        if os.path.isfile(os.path.join(args.out_dir, window_file)): 
            data = pd.read_csv(os.path.join(args.out_dir, window_file), sep = ',')
        else: 
            ## get window encodings
            if args.type == 'HLA':
                data = pd.read_csv(file, sep = ',')
                if 'Group' in data.columns:
                    data = data[['Hit', 'Epitope', 'Sequence', 'Group']]
                else:
                    data = data[['Hit', 'Epitope', 'Sequence']]
                tqdm.pandas(desc = 'Creating window encodings', leave = False)
                data['Encoded'] = data.progress_apply(lambda row: get_window_encodings(row['Epitope'], row['Sequence'], aa_score_dict, -1, args.l), axis = 1)
            elif args.type == 'MUTINT':
                data = pd.read_csv(file, sep = ',')
                if 'Group' in data.columns:
                    data = data[['Y2H_score', 'Position', 'Data', 'Mutated_Seq (unless WT)', 'Interactor_Seq', 'Group']]
                else:
                    data = data[['Y2H_score', 'Position', 'Data', 'Mutated_Seq (unless WT)', 'Interactor_Seq']]
                tqdm.pandas(desc = 'Creating window encodings', leave = False)
                data['Encoded'] = data.progress_apply(lambda row: get_window_encodings(row['Mutated_Seq (unless WT)'], row['Interactor_Seq'], aa_score_dict, row['Position'], args.l), axis = 1)
            else: 
                sys.exit('Unsupported data files. Please try again.')
            data.to_csv(os.path.join(args.out_dir, 'window_encodings', window_file), index = False)
        tqdm.pandas(desc = 'Creating k-mers', leave = False)
        data['k-mers'] = data.progress_apply(lambda row: get_kmers_str(row['Encoded'], args.k, args.sub_size), axis = 1)
        
        ## extract unique k-mers
        kmers = [i for i in data['k-mers']]

        ## extract encoded sequences
        encoded = [i for i in data['Encoded']]

        ## write k-mers to file
        output_file = '{0}_vocab_k{1}_subsize{2}.txt'.format(data_name, args.k, args.sub_size)        
        with open(os.path.join(args.out_dir, 'kmers_txt', output_file), 'w+') as out_file:
            out_file.write('\n'.join(kmers))

        ## write encoded sequences to file
        with open(os.path.join(args.out_dir, 'encoded_txt', output_file), 'w+') as out_file:  
            out_file.write('\n'.join(encoded))
  
  
        output_file = '{0}_vocab_k{1}_subsize{2}.csv'.format(data_name, args.k, args.sub_size)   
        data.to_csv(os.path.join(args.out_dir, 'kmers_csv', output_file), index = False)


        if args.freq > 0:
            ## find k-mers to keep (by frequency)
            all_kmers = ' '.join(kmers)
            kmer_list = all_kmers.split(' ')
            counts = Counter(kmer_list)
            if args.freq < 1:
                threshold = np.quantile(list(counts.values()), args.freq)
            else:
                threshold = args.freq
            freq_kmers = { x: count for x, count in counts.items() if count >= threshold }

            ## filter k-mers 
            new_df = pd.DataFrame(columns = data.columns.tolist())
            filtered_kmers = []
            for idx, row in data.iterrows():
                kmer = row['k-mers']
                kmer_split = kmer.split(' ')
                filtered_kmer = [x for x in kmer_split if x in freq_kmers]
                row['k-mers'] = ' '.join(filtered_kmer)
                filtered_kmers.append(' '.join(filtered_kmer))
                new_df.loc[idx] = row

            ## write to file
            output_file = '{0}_vocab_k{1}_subsize{2}_filt.txt'.format(data_name, args.k, args.sub_size)        
            with open(os.path.join(args.out_dir, 'kmers_txt', 'filt', output_file), 'w+') as out_file:
                out_file.write('\n'.join(filtered_kmers))

            output_file = '{0}_vocab_k{1}_subsize{2}_filt.csv'.format(data_name, args.k, args.sub_size)     
            new_df.to_csv(os.path.join(args.out_dir, 'kmers_csv', 'filt', output_file), index = False)

                      
def build_aa_score_dict(grantham_scores_dict: dict):
    ## Builds the amino acid score dictionary by finding the difference between every possible pair of 
    ## elements in the vocabulary of amino acids.
    ## REQUIRES: none
    ## MODIFIES: aa_score_dict - populates this dictionary with the score differences for every
    ##                           possible pair of amino acids
    ## RETURNS:  none
    grantham_keys = list(grantham_scores_dict.keys())
    aa_score_dict = {}
    for i in range(len(grantham_scores_dict)):
        ## get unique pairs of the AAs
        for j in range(len(grantham_scores_dict) - i): 
            aa_pair = grantham_keys[i] + grantham_keys[j + i] 
            ## get the abs value of unique pairs' differences, rounded
            aa_pair_score = round(abs(grantham_scores_dict[grantham_keys[i]] - grantham_scores_dict[grantham_keys[j + i]]))
            aa_score_dict[aa_pair] = aa_pair_score # forward
            aa_score_dict[aa_pair[::-1]] = aa_pair_score # and reverse
    return aa_score_dict

                      
def get_kmers_str(encoding_scores: str, k: int = 5, sub_size: int = 1):
    ## Create the sequence of k-mers from the encoding scores 
    ## REQUIRES: encoding_scores - a string; the scores calculated using get_window_encodings()
    ##           k - an integer; the number of 'chunks' to use in each 'word' (default: 5)          
    ##           sub_size - an integer; number of characters to skip when creating k-mers (skip_size number of characters will not overlap between two adjacent k-mers) (default: 1)
    ## MODIFIES: none
    ## RETURNS:  a string; the k-mers
    
    padding = {'9'}
    for i in range(k):
        padding.add(str(9) * (i + 1))

    ## ppi-specific k-mers
    kmers = []
    j = 0
    while j < len(encoding_scores):
        kmer = encoding_scores[j:j+k]
        if kmer not in padding:
            kmers.append(kmer)
        j = j + sub_size
    kmers = ' '.join(kmers)
    return kmers


def get_window_encodings(mut_window: str, interactor: str, aa_score_dict: dict, pos: int = -1, l: int = -1):
    ## Create the window encodings by sliding the epitope window along the interactor sequence.
    ## REQUIRES: mut_window - a string; the amino acid sequence for the binding epitope
    ##           interactor - a string; the amino acid sequence for the MHC
    ##           aa_score_dict - a dictionary; the scores for each amino acid pair
    ##           pos - an integer; the position at which the mutation occurs (OPTIONAL)
    ##           l - an integer; the number of amino acids to use on each side of pos to create the sliding window
    ## MODIFIES: none
    ## RETURNS:  a string; the encoding scores using SWING
    ppi_encoding = ''
    ## for sliding window
    its = 0 
    if pos >= 0 & l >= 0:
        ## -1 because python is 0 indexed
        pos = pos - 1
        ## sliding mutant window across entire interactor - l AAs on each side of mutation position
        start = max(pos - l, 0)
        end = pos + 1 + l
        mut_window_sub = mut_window[start:end]
    else:
        mut_window_sub = mut_window

    for j in range(len(interactor)): 
        window_scores = ''
        ## at each positon of the interactor, align mutant window and find the score differences 
        for k in range(len(mut_window_sub)): 
            try: # no directionality
                pair = mut_window_sub[k]+interactor[k+its]
                score = aa_score_dict[pair]                
            except: # if not a pair, it is padding (have reached the end of the interactor)
                pair = None
                score = 9
            window_scores = window_scores + str(score) # string per mut window
        its += 1 # sliding down to next position on the interactor
        # if str(window_scores) == '9' * len(mut_window_sub):
        #     break
        ppi_encoding = ppi_encoding + str(window_scores) # final string per interaction
    return ppi_encoding


if __name__ == "__main__":
    main()


### SUMMER 2023 IMPLEMENTATION
# def main():
#     parser = argparse.ArgumentParser(prog = 'Vocabulary File Builder.',
#                                      description = 'Read datasets and create vocabulary files.')
#     parser.add_argument('-d', '--data_dir', help = 'path to directory containing datasets')
#     parser.add_argument('-o', '--out_dir', help = 'path to directory in which to save output files')
#     parser.add_argument('-k', '--k', type = int, default = 2, help = 'number of chunks in a k-mer')
#     parser.add_argument('-s', '--sub_size', type = int, default = 3 , help = 'number of characters in a chunk')
#     parser.add_argument('-l', '--l', type = int, default = 2, help = 'number of positions on either side of mutation position for sliding window')
#     parser.add_argument('-t', '--type', type = str, default = 'HLA', dest = 'type', help = 'the type of dataset')
#     args = parser.parse_args()
    
#     ## create directory to save output if it doesn't exist
#     if not os.path.exists(args.out_dir):
#         os.makedirs(args.out_dir)
#     if not os.path.exists(os.path.join(args.out_dir, 'kmers_csv')):    
#         os.makedirs(os.path.join(args.out_dir, 'kmers_csv'))
#     if not os.path.exists(os.path.join(args.out_dir, 'window_encodings')):
#         os.makedirs(os.path.join(args.out_dir, 'window_encodings'))
#     if not os.path.exists(os.path.join(args.out_dir, 'kmers_txt')):
#         os.makedirs(os.path.join(args.out_dir, 'kmers_txt'))
    
#     ## get list of data files
#     if args.data_dir.endswith('.csv'):
#         data_files = [args.data_dir]
#     else:
#         data_files = list(pathlib.Path(args.data_dir).glob('*.csv'))
    
#     ## build amino acid grantham score dictionary
#     grantham_scores_dict = {'A': 8.1,  'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5,
#                             'E': 12.3, 'Q': 10.5, 'G': 9.0,  'H': 10.4, 'I': 5.2,
#                             'L': 4.9,  'K': 11.3, 'M': 5.7,  'F': 5.2,  'P': 8.0,
#                             'S': 9.2,  'T': 8.6,  'W': 5.4,  'Y': 6.2,  'V': 5.9}
#     aa_score_dict = build_aa_score_dict(grantham_scores_dict)

#     ## iterate through data files
#     for file in tqdm(data_files, desc = 'Reading file', position = 1, leave = True):
#         data_name = str(file).rsplit('/', 1)[-1].rsplit('.csv', 1)[0]
#         print(data_name)
#         window_file = '{0}_vocab_window_encodings.csv'.format(data_name)
#         ## check if window encodings have already been created --- this is the most intensive step
#         if os.path.isfile(os.path.join(args.out_dir, window_file)): 
#             data = pd.read_csv(os.path.join(args.out_dir, window_file), sep = ',')
#         else: 
#             ## get window encodings
#             if args.type == 'HLA':
#                 data = pd.read_csv(file, sep = ',')[['Hit', 'Epitope', 'Sequence']]
#                 tqdm.pandas(desc = 'Creating window encodings', leave = False)
#                 data['Encoded'] = data.progress_apply(lambda row: get_window_encodings(row['Epitope'], row['Sequence'], aa_score_dict, -1, args.l), axis = 1)
#             elif args.type == 'MUTINT':
#                 data = pd.read_csv(file, sep = ',')[['Y2H_score', 'Position', 'Data', 'Mutated_Seq (unless WT)', 'Interactor_Seq']]
#                 tqdm.pandas(desc = 'Creating window encodings', leave = False)
#                 data['Encoded'] = data.progress_apply(lambda row: get_window_encodings(row['Mutated_Seq (unless WT)'], row['Interactor_Seq'], aa_score_dict, row['Position'], args.l), axis = 1)
#             else: 
#                 sys.exit('Unsupported data files. Please try again.')
#             data.to_csv(os.path.join(args.out_dir, 'window_encodings', window_file), index = False)
#         tqdm.pandas(desc = 'Creating k-mers', leave = False)
#         data['k-mers'] = data.progress_apply(lambda row: get_kmers_str(row['Encoded'], args.k, args.skip_size), axis = 1)
#         print(data['k-mers'])
        
#         ## extract unique k-mers
#         kmers = [i for i in data['k-mers']]
        
#         ## write to file
#         output_file = '{0}_vocab_k{1}_subsize{2}.txt'.format(data_name, args.k, args.sub_size)        
#         with open(os.path.join(args.out_dir, 'kmers_txt', output_file), 'w+') as out_file:
#             out_file.write('\n'.join(kmers))

#         output_file = '{0}_vocab_k{1}_subsize{2}.csv'.format(data_name, args.k, args.sub_size)     
#         data.to_csv(os.path.join(args.out_dir, 'kmers_csv', output_file), index = False)

# def get_kmers_str(encoding_scores: str, k: int = 2, sub_size: int = 3):
#     ## Create the sequence of k-mers from the encoding scores 
#     ## REQUIRES: encoding_scores - a string; the scores calculated using get_window_encodings()
#     ##           k - an integer; the number of 'chunks' to use in each 'word' (default: 2)
#     ##           sub_size - an integer; the length of the 'chunks' (default: 3)
#     ## MODIFIES: none
#     ## RETURNS:  a string; the k-mers
    
#     padding = {'9'}
#     for i in range(k * sub_size):
#         padding.add(str(9) * (i + 1))

#     kmers = []
#     ## split encoding scores into subsequences of length subsize
#     split_enc = [encoding_scores[i:i+sub_size] for i in range(0, len(encoding_scores), sub_size) if encoding_scores[i:i+sub_size] != '9' * sub_size]

#     ## check if last subsequence is of the correct length
#     ## if it is just '9's, then throw it away
#     ## if not, pad with '9' to proper length
#     last_subseq = split_enc.pop()
#     if len(last_subseq) < sub_size:
#         if not last_subseq == '9' * len(last_subseq):
#             last_subseq = last_subseq.ljust(sub_size, '9')
#             split_enc.append(last_subseq)

#     for j in range(0, len(split_enc), k): # keep padding in k-mers?
#         kmer = ''.join(split_enc[j:j+k])
#         if kmer in padding:
#             pass
#         else:
#             kmer = kmer.ljust(k * sub_size, '9')
#             kmers.append(kmer) # overlapping k-mers

#     kmers = ' '.join(kmers)
#     return kmers

# def get_window_encodings(mut_window: str, interactor: str, aa_score_dict: dict, pos: int = -1, l: int = -1):
#     ## Create the window encodings by sliding the epitope window along the interactor sequence.
#     ## REQUIRES: mut_window - a string; the amino acid sequence for the binding epitope
#     ##           interactor - a string; the amino acid sequence for the MHC
#     ##           aa_score_dict - a dictionary; the scores for each amino acid pair
#     ##           pos - an integer; the position at which the mutation occurs (OPTIONAL)
#     ##           l - an integer; the number of amino acids to use on each side of pos to create the sliding window
#     ## MODIFIES: none
#     ## RETURNS:  a string; the encoding scores using SWING
#     ppi_encoding = ''
#     ## for sliding window
#     its = 0 
#     if pos >= 0 & l >= 0:
#         ## -1 because python is 0 indexed
#         pos = pos - 1
#         ## sliding mutant window across entire interactor - l AAs on each side of mutation position
#         start = max(pos - l, 0)
#         end = pos + 1 + l
#         mut_window_sub = mut_window[start:end]
#     else:
#         mut_window_sub = mut_window
    
#     for i in range(len(interactor)): 
#         window_scores = ''
#         ## at each positon of the interactor, align mutant window and find the score differences 
#         for i in range(len(mut_window_sub)): 
#             try: # no directionality
#                 pair = mut_window_sub[i]+interactor[i+its]
#                 try: # get the score
#                     score = aa_score_dict[pair]
#                 except: # no directionality- flip the AA pair if not in dictionary to get score
#                     pair = pair[::-1] # reverse string
#                     score = aa_score_dict[pair]
#             except: # if not a pair, it is padding (have reached the end of the interactor)
#                 pair = None
#                 score = 9
#             window_scores = window_scores + str(score) # string per mut window

#         its += 1 # sliding down to next position on the interactor
#         if str(window_scores) == '9' * len(mut_window_sub):
#             break
#         ppi_encoding = ppi_encoding + str(window_scores) # final string per interaction
#     return ppi_encoding

         
