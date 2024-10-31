import pandas as pd  # For data handling
import numpy as np
import os
import random
random.seed(42)
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import sklearn
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser('Cross Prediction of  MHC:Peptide Binding.')
parser.add_argument('--data_set',required=True,help='data to train with',default="data.csv")
parser.add_argument('--output',required=True,help='output prefix',default='1')
parser.add_argument('--cross_pred_set',required=True,help='The prefix for the cross pred dataset',default='HLA-B35:03')
#Language parameters
parser.add_argument('--w',default=7,type=int,help='how large the surrounding window is')
parser.add_argument('--k',default=7,type=int,help='size of words (kmer) for the window encoding to be fragmented into')
parser.add_argument('--metric',required=False,help='The biochemical metric for calculating the scores between AA. Options: ["polarity","hydrophobicity"]',default="polarity")
parser.add_argument('--padding_score',default=9,type=int,help='The value for padding. Should be outside the range of delta biochemical metric score.')
#Doc2Vec parameters
parser.add_argument('--dm',default=0,type=int,help='fill dm info')
parser.add_argument('--dim',default=64,type=int,help='fill dim')
parser.add_argument('--epochs',default=40,type=int,help='number epochs')
parser.add_argument('--min_count',default=1,type=int,help='fill minimum count')
parser.add_argument('--alpha',default=0.42,type=float,help='alpha value')
parser.add_argument('--save_embeddings',default=True, type=bool, help='Option to save Doc2Vec embeddings, to run cross-prediction with other seeds.')
#XGBoost
parser.add_argument('--n_estimators',default=200,type=int,help='fill n estimators')
parser.add_argument('--max_depth',default=1,type=int,help='maximum dempth')
parser.add_argument('--learning_rate',default=0.71,type=float,help='learning rate')

args = parser.parse_args()

#creating output directory
os.makedirs("output", exist_ok=True)
os.makedirs("output/cross_pred", exist_ok=True)
os.makedirs("output/models", exist_ok=True)

# creating amino acid vocabulary
if args.metric == 'polarity':
        AA_scores = {'A':8.1,'R':10.5,'N':11.6,'D':13.0,'C':5.5,'E':12.3,'Q':10.5,'G':9.0,'H':10.4,'I':5.2,
                     'L':4.9,'K':11.3,'M':5.7,'F':5.2,'P':8.0,'S':9.2,'T':8.6,'W':5.4,'Y':6.2,'V':5.9} # grantham scores
if args.metric == 'hydrophobicity':
        AA_scores = {'A': 5.33, 'R': 4.18, 'D': 3.59, 'N': 3.59, 'C': 7.93, 'Q': 3.87, 'E': 3.65, 'G': 4.48, 'H': 5.1, 'I': 8.83,
                     'L': 8.47, 'K': 2.95, 'M': 8.95, 'F': 9.03, 'P': 3.87, 'S': 4.09, 'T': 4.49, 'W': 7.66, 'Y': 5.89, 'V': 7.63} #Miyazawa S Hydrophobicity scale 

AAs = list(AA_scores.keys())

aa_score_dict = {} # key: AA pair, value: grantham score difference
for i in range(len(AAs)): # create all pairs of AAs
    for j in range(len(AAs)-i):
        AA_pair = AAs[i]+AAs[j+i]
        AA_pair_score = round(abs(AA_scores[AAs[i]]-AA_scores[AAs[j+i]])) # take rounded, absolute value of the difference of scores
        aa_score_dict[AA_pair] = AA_pair_score # forward
        aa_score_dict[AA_pair[::-1]] = AA_pair_score # and reverse
            
def get_window_encodings(df): # takes df (mut/int sequences and mutation position) and window k (k AA's on each side of the mutation position)
    """
    Takes a pandas dataframe where each row represents a protein-protein/peptide-protein interaction.  
  
    Customization includes setting the interactor protein and the peptide window. In the pMHC context, the epitope defines the peptide window. In the missense mutation pertubation context, the window_k parameter defines the size of the window and the mutation defines the position. Additionally, the scale used to calculate the score can be altered. If the scale is changed the padding_score may need to be adjusted.  
  
    The function returns a list of score encodings strings that each represent a PPI. The ends of the encodings include padding from the sliding window process. These encodings will be broken into k-mers for the embedding model.
    """

    total_encodings = [] # final list of encodings
    
    for i in (df.index): # iterate through all pairs
        mut_window = df['Epitope'].iloc[i]
        interactor = df['Sequence'].iloc[i] 
        
        PPI_encoding = '' # for each PPI, dealing with strings
        its = 0 # for sliding window
        for j in range(len(interactor)): # sliding mutant window across entire interactor
            
            window_scores = ''
            for k in range(len(mut_window)): # at each positon of the interactor, align mutant window and find the score differences 
                try: # no directionality
                    pair = mut_window[k]+interactor[k+its]
                    
                    try: # get the score
                        score = aa_score_dict[pair]
                    
                    except: # no directionality- flip the AA pair if not in dictionary to get score
                        pair = pair[::-1] # reverse string
                        score = aa_score_dict[pair]
                
                except: # if not a pair, it is padding (have reached the end of the interactor)
                    pair = None
                    score = 9 # padding
                window_scores = window_scores + str(score) # string per mut window
                
            its +=1 # sliding down to next position on the interactor
            PPI_encoding = PPI_encoding + str(window_scores) # final string per interaction
            
        total_encodings.append(PPI_encoding) # all strings for all interactions
        
    return total_encodings

def get_kmers_str(encoding_scores,k=7,shuffle=False, padding_score=9):
    """
    Takes the encoding scores from get_window_encodings().  
  
    Customization includes setting size of the kmers (k), a shuffle option, and the integer defining the padding score.  
  
    This function returns a list of lists of overlapping k-mers of specified size k, removing k-mers of only padding. Each list of k-mers are specific to each of the PPIs. This output is compatible with gensims
    """

    padding = {str(padding_score)}

    for i in range(k):
        padding.add(str(padding_score)*(i+1))
    kmers = []
    for ppi_score in encoding_scores:
        int_kmers = []
        for j in range(len(ppi_score)): # remove padding only kmers.
            kmer = ppi_score[j:j+k]
            if kmer in padding:
                pass
            else:
                int_kmers.append(kmer) # overlapping k-mers 
        if shuffle:
            random.shuffle(int_kmers)
        kmers.append(int_kmers)
    return kmers

def get_corpus(matrix, tokens_only=False): # turns each ppi into a d2v tagged document
    for i in range(len(matrix)):
        yield gensim.models.doc2vec.TaggedDocument(matrix[i],[i])


k = args.k
dim = args.dim
dm = args.dm
w = args.w
min_count = args.min_count
alpha = args.alpha
epochs = args.epochs
output = args.output
cross_pred_set = args.cross_pred_set
padding_score = args.padding_score

n_estimators = args.n_estimators
max_depth = args.max_depth
learning_rate = args.learning_rate

df = pd.read_csv(args.data_set)
df = df.sample(frac = 1, random_state = 1).reset_index()

window_encodings = get_window_encodings(df)
initial_kmers = get_kmers_str(window_encodings,k=k,shuffle=False) # no shuffle

kmers = []

for i in range(len(initial_kmers)):
    if i in set(df[df['Set']=='Test'].index):
        kmers.append(initial_kmers[i])
        #print(initial_kmers[i])

    else:
        random.shuffle(initial_kmers[i])
        kmers.append(initial_kmers[i]) # shuffle just the train set    
        
train_corpus = list(get_corpus(kmers))

d2v_model = Doc2Vec(vector_size=dim,dm=dm,window=w,min_count=1,alpha=alpha)
d2v_model.build_vocab(train_corpus)
d2v_model.train(train_corpus,total_examples=d2v_model.corpus_count, epochs=epochs)

all_vecs = d2v_model.dv.vectors

df["Vectors"] = all_vecs.tolist()

if args.save_embeddings==True:
        print("Saving Doc2Vec model...")
        d2v_model.save("output/models/doc2vec_nolabels_{0}.model".format(output))
        df.to_csv('output/models/{0}_shuffled_df_nolabels_vectors.csv'.format(output), index=False)

tr_vecs = []
predict_vecs = []
for j in range(len(all_vecs)):
    if j in set(df[df['Set']=='Test'].index):
        predict_vecs.append(all_vecs[j])
    else:
        tr_vecs.append(all_vecs[j])

tr_y = df[df['Set']!='Test'].Hit.values.reshape(-1,1)

# train xbg on all the 'data'
xgb_cl = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
print(len(tr_vecs))
print(len(tr_y))

xgb_cl.fit(tr_vecs,tr_y)
    
y_hat = xgb_cl.predict(predict_vecs)
y_hat_proba = xgb_cl.predict_proba(predict_vecs)[::,1]

denovo_prediction = pd.DataFrame()
denovo_prediction['Epitope'] = df[df['Set']=='Test']['Epitope']
denovo_prediction['MHC'] = df[df['Set']=='Test']['MHC']
denovo_prediction['Predicted Y'] =  y_hat
denovo_prediction['Predicted Probabilities Y'] = y_hat_proba 
denovo_prediction = denovo_prediction.sort_values(by='Predicted Probabilities Y', ascending=False)
denovo_prediction.to_csv('output/cross_pred/{0}_predictions_only.csv'.format(output), index=False)


