import pandas as pd # For data handling
import numpy as np
import os
import random
random.seed(42)
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import sklearn
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, auc
import pickle
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser('No label prediction of missense mutation perturbation on specific PPIs.')
parser.add_argument('--data_set',required=True,help='data to train with',default="../Data/MutInt_Model/Mutation_perturbation_model.csv")
parser.add_argument('--output',required=True,help='output prefix',default='nolabel_preds')
parser.add_argument('--nolabel_pred_set',required=True,help='The prefix for the cross pred dataset',default='nolabel_preds')
#Language parameters
parser.add_argument('--k',default=7,type=int,help='size of words (kmer) for the window encoding to be fragmented into')
parser.add_argument('--L',default=1,type=int,help='the number of amino acids considered around the mutation for the sliding window')
parser.add_argument('--metric',required=False,help='The biochemical metric for calculating the scores between AA. Options: ["polarity","hydrophobicity"]',default="polarity")
parser.add_argument('--padding_score',default=9,type=int,help='The value for padding. Should be outside the range of delta biochemical metric score.')
#Doc2Vec parameters
parser.add_argument('--w',default=6,type=int,help='how large the surrounding window is')
parser.add_argument('--dm',default=1,type=int,help='fill dm info')
parser.add_argument('--dim',default=128,type=int,help='fill dim')
parser.add_argument('--epochs',default=52,type=int,help='number epochs')
parser.add_argument('--min_count',default=1,type=int,help='fill minimum count')
parser.add_argument('--alpha',default=0.08711,type=float,help='alpha value')
parser.add_argument('--save_embeddings',default=True, type=bool, help='Option to save Doc2Vec embeddings, to run cross-prediction with other seeds.')
#XGBoost
parser.add_argument('--n_estimators',default=375,type=int,help='fill n estimators')
parser.add_argument('--max_depth',default=6,type=int,help='maximum dempth')
parser.add_argument('--learning_rate',default=0.08966,type=float,help='learning rate')

args = parser.parse_args()

#creating output directory
os.makedirs("output", exist_ok=True)
os.makedirs("output/nolabel_pred", exist_ok=True)
os.makedirs("output/models", exist_ok=True)

# creating amino acid vocabulary
if args.metric == 'polarity':
        AA_scores = {'A':8.1,'R':10.5,'N':11.6,'D':13.0,'C':5.5,'E':12.3,'Q':10.5,'G':9.0,'H':10.4,'I':5.2,
                     'L':4.9,'K':11.3,'M':5.7,'F':5.2,'P':8.0,'S':9.2,'T':8.6,'W':5.4,'Y':6.2,'V':5.9} # grantham scores
if args.metric == 'hydrophobicity':
        AA_scores = {'A': 5.33, 'R': 4.18, 'D': 3.59, 'N': 3.59, 'C': 7.93, 'Q': 3.87, 'E': 3.65, 'G': 4.48, 'H': 5.1, 'I': 8.83,
                     'L': 8.47, 'K': 2.95, 'M': 8.95, 'F': 9.03, 'P': 3.87, 'S': 4.09, 'T': 4.49, 'W': 7.66, 'Y': 5.89, 'V': 7.63} #Miyazawa S Hydrophobicity scale 

AAs = list(AA_scores.keys())
aa_score_dict = {} 
for i in range(len(AAs)): # create all pairs of AAs
    for j in range(len(AAs)-i):
        AA_pair = AAs[i]+AAs[j+i]
        AA_pair_score = round(abs(AA_scores[AAs[i]]-AA_scores[AAs[j+i]])) # take rounded, absolute value of the difference of scores
        aa_score_dict[AA_pair] = AA_pair_score # forward
        aa_score_dict[AA_pair[::-1]] = AA_pair_score # and reverse
        
def get_window_encodings(df, window_k=1, pos_colname='Position', mutseq_colname='Mutated_Seq (unless WT)', intseq_colname='Interactor_Seq', aa_score_dict=aa_score_dict, padding_score=9): # Takes df (mut/int sequences and mutation position) and window_k (# AA's on each side of the mutation position)
    total_encodings = [] # Master list of encodings
    for i in tqdm(df.index): # Iterate through protein pairs
        pos = df[pos_colname].iloc[i]-1 # find mutation position for window
        mut_window = df[mutseq_colname].iloc[i][pos-window_k:pos+window_k+1] # Create sliding window
        interactor = df[intseq_colname].iloc[i] # Get interactor sequence
        PPI_encoding = '' # For each PPI
        its = 0 # Tracks sliding window position
        for j in range(len(interactor)): # For the entire length of the interactor
            window_scores = '' # Saves the scores between window-interactor at the 'its' position
            for k in range(len(mut_window)): # At each positon of the interactor ('its'), align mutant window and find the score differences
                try: # If 'its' is at the end of the interactor, the window is hanging off end (padding)
                    pair = mut_window[k]+interactor[k+its] 
                    score = aa_score_dict[pair]
                except: # If not a pair, its padding (end of interactor)
                    pair = None
                    score = padding_score # Padding score is 9
                window_scores = window_scores + str(score) # Add score to running string
            its +=1 # Slide down a position on the interactor
            PPI_encoding = PPI_encoding + str(window_scores) # Add to final string for interaction
        total_encodings.append(PPI_encoding) # Add to list for all interactions
    return total_encodings # List of encodings for each PPI

def get_kmers_str(encoding_scores, k=7, padding_score=9):
    padding = {str(padding_score)} 
    for i in range(k): # Makes a set of padding scores that will be removed from the final k-mers
        padding.add(str(padding_score)*(i+1)) # {'9','99','999'...}
    kmers = [] # Master list of k-mers
    for ppi_score in tqdm(encoding_scores): # For each PPI encoding
        int_kmers = [] # K-mers specific to PPI
        for j in range(len(ppi_score)): # Iterate over the PPI encoding
            kmer = ppi_score[j:j+k] # Slice k-mers and sliding over
            if kmer in padding: # If K-mer is just padding, don't add it
                pass
            else:
                int_kmers.append(kmer) # Keep non-padding k-mers  
        kmers.append(int_kmers) # Append k-mers to master list
    return kmers 

def get_corpus(matrix, tokens_only=False):
    for i in range(len(matrix)): # for each PPI
        yield gensim.models.doc2vec.TaggedDocument(matrix[i],[i]) # Create a tagged document

output = args.output

k = args.k
L = args.L
padding_score = args.padding_score

w = args.w
dm = args.dm
dim = args.dim
epochs = args.epochs
min_count = args.min_count
alpha = args.alpha

n_estimators = args.n_estimators
max_depth = args.max_depth
learning_rate = args.learning_rate

# load in data (should be mutant only!)
df = pd.read_csv(args.data_set)

# add in wild types
test_muts = df[df['Set']=='Test'] # subset the test mutants
wt_seqs = []
for i in test_muts.index: # for each mutant
    # change the mutant sequence back to wild type
    mut_seq = test_muts.loc[i]['Mutated_Seq (unless WT)']
    before_aa = test_muts.loc[i]['Before_AA']
    after_aa = test_muts.loc[i]['After_AA']
    position = test_muts.loc[i]['Position'] - 1 # POSITION IS ONE INDEXED
    test_muts.loc[i]['Mutated_Seq (unless WT)']
    if mut_seq[position] == after_aa: # check if after_AA is really at the position
        wt_seqs.append(mut_seq[:position]+after_aa+mut_seq[position+1:]) # make the wt sequence
    else:
        raise ValueError('Position Index (1 indexed) does not match Before_AA')

# add WT nolabels to df and shuffle
test_wts = test_muts.copy()
test_wts['Mutated_Seq (unless WT)']=wt_seqs
test_wts['Type']='WildType'
df = pd.concat([df, test_wts]).sample(frac = 1, random_state = 1).reset_index(drop=True) # shuffle

# encode and k-merize
window_encodings = get_window_encodings(df, window_k=L, pos_colname='Position', mutseq_colname='Mutated_Seq (unless WT)', intseq_colname='Interactor_Seq', aa_score_dict=aa_score_dict, padding_score=padding_score)
kmers = get_kmers_str(window_encodings, k=k, padding_score=padding_score)
train_corpus = list(get_corpus(kmers))

# train D2V (embedding everything together)
d2v_model = Doc2Vec(vector_size=dim, min_count=min_count, alpha=alpha, dm=dm, window=w)
d2v_model.build_vocab(train_corpus)
d2v_model.train(train_corpus,total_examples=d2v_model.corpus_count, epochs=epochs)

# save vectors
all_vecs = d2v_model.dv.vectors
df["Vectors"] = all_vecs.tolist()

if args.save_embeddings==True:
        print("Saving Doc2Vec model...")
        d2v_model.save("output/models/doc2vec_nolabels_{0}.model".format(output))
        df.to_csv('output/models/{0}_shuffled_df_nolabels_vectors.csv'.format(output), index=False)

# separate training and testing vectors
tr_vecs = []
predict_vecs = []
for i in range(len(all_vecs)):
    if i in set(df[(df['Set']=='Test') & (df['Type']=='Mutant')].index): # just mutants
        predict_vecs.append(all_vecs[i])
    elif i in set(df[df['Set']!='Test'].index): # training set
        tr_vecs.append(all_vecs[i])
    else: # ignoring no label wild types
        pass 

tr_y = df[df['Set']!='Test'].Y2H_score.values.reshape(-1,1)

# train xbg on all the data
xgb_cl = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
xgb_cl.fit(tr_vecs,tr_y)
    
y_hat = xgb_cl.predict(predict_vecs)
y_hat_proba = xgb_cl.predict_proba(predict_vecs)[::,1]

denovo_prediction = pd.DataFrame()
denovo_prediction['Mutated_Seq'] = df[(df['Set']=='Test') & (df['Type']=='Mutant')]['Mutated_Seq (unless WT)']
denovo_prediction['Interactor_Seq'] = df[(df['Set']=='Test') & (df['Type']=='Mutant')]['Interactor_Seq']
denovo_prediction['Before_AA'] = df[(df['Set']=='Test') & (df['Type']=='Mutant')]['Before_AA']
denovo_prediction['Position'] = df[(df['Set']=='Test') & (df['Type']=='Mutant')]['Position']
denovo_prediction['After_AA'] = df[(df['Set']=='Test') & (df['Type']=='Mutant')]['After_AA']

denovo_prediction['Predicted Y'] = y_hat
denovo_prediction['Predicted Probabilities Y'] = y_hat_proba 
denovo_prediction = denovo_prediction.sort_values(by='Predicted Probabilities Y', ascending=False)
denovo_prediction.to_csv('output/nolabel_pred/{0}_predictions_only.csv'.format(output), index=False)
