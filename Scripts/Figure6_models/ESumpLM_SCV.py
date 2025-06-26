# EmbedSum SCV
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, auc
import argparse
import wandb
from imblearn.over_sampling import RandomOverSampler, SMOTE , ADASYN ,SVMSMOTE, BorderlineSMOTE
from collections import Counter
from tqdm import tqdm
import pickle

# Your Data Here
df = pd.read_csv('../../Data/MutInt_Model/Mutation_perturbation_model.csv')

# Language Generation
AA_scores = {'A':8.1,'R':10.5,'N':11.6,'D':13.0,'C':5.5,'E':12.3,'Q':10.5,'G':9.0,'H':10.4,'I':5.2,
            'L':4.9,'K':11.3,'M':5.7,'F':5.2,'P':8.0,'S':9.2,'T':8.6,'W':5.4,'Y':6.2,'V':5.9} # Grantham score, interchangable
AAs = list(AA_scores.keys())
aa_score_dict = {} 
for i in range(len(AAs)): # create all pairs of AAs
    for j in range(len(AAs)-i):
        AA_pair = AAs[i]+AAs[j+i]
        AA_pair_score = round(abs(AA_scores[AAs[i]]-AA_scores[AAs[j+i]])) # take rounded, absolute value of the difference of scores
        aa_score_dict[AA_pair] = AA_pair_score # forward
        aa_score_dict[AA_pair[::-1]] = AA_pair_score # and reverse

# Functions
def get_embsum_encoding(df, mutseq_colname='Mutated_Seq (unless WT)', intseq_colname='Interactor_Seq', AA_scores_dict=AA_scores): # Takes df (mut/int sequences and mutation position) and window_k (# AA's on each side of the mutation position)
    mut_encodings = [] # Master list of encodings
    int_encodings = [] # Master list of encodings
    for i in tqdm(df.index): # Iterate through protein pairs
        mutated_seq = df[mutseq_colname].iloc[i] # Get mutated sequence
        mut_encoding = ''
        for m in mutated_seq: # Create encoding for mutant sequence
            mut_encoding = mut_encoding + str(round(AA_scores_dict[m])) + ' '
        mut_encodings.append(mut_encoding[:-1]) # Add to list for all interactions, slice off last space
        
        interactor_seq = df[intseq_colname].iloc[i] # Get interactor sequence
        int_encoding = ''
        for t in interactor_seq: # Create encoding for interactor sequence
            int_encoding = int_encoding + str(round(AA_scores_dict[t])) + ' '
        int_encodings.append(int_encoding[:-1]) # Add to list for all interactions, slice off last space
    return mut_encodings, int_encodings # Lists of encodings of mutant and interactor proteins for each PPI
    
def get_kmers_embsum(encoding_scores, k=7): 
    kmers = [] # Master list of k-mers
    for ppi_score in tqdm(encoding_scores): # For each PPI encoding
        int_kmers = [] # K-mers specific to PPI
        score_list = ppi_score.split()
        for j in range(len(score_list)-k+1): # Iterate over the PPI encoding
            kmer = " ".join(score_list[j:j+k]) # make k-mers and sliding over
            int_kmers.append(kmer) # Keep non-padding k-mers  
        kmers.append(int_kmers) # Append k-mers to master list
    return kmers

def get_corpus(matrix, tokens_only=False):
    for i in range(len(matrix)): # for each PPI
        yield gensim.models.doc2vec.TaggedDocument(matrix[i],[i]) # Create a tagged document

# Best EmbedSum params for MutInt SCV
k1 = 7
k2 = 2

dim1 = 128 # fixed
dm1 = 0
alpha1 = 0.060215
w1 = 1
epochs1 = 197

dim2 = 128 # fixed
dm2 = 1
alpha2 = 0.00052088
w2 = 4
epochs2 = 121

n_estimators = 191
max_depth = 7
learning_rate = 0.057252

# Process mutant and interactor sequences SEPERATELY
mut_encodings, int_encodings = get_embsum_encoding(df, mutseq_colname='Mutated_Seq (unless WT)', intseq_colname='Interactor_Seq', AA_scores_dict=AA_scores)
mut_kmers = get_kmers_embsum(mut_encodings, k=k1) # mut is 1
int_kmers = get_kmers_embsum(int_encodings, k=k2) # int is 2
mut_train_corpus = list(get_corpus(mut_kmers))
int_train_corpus = list(get_corpus(int_kmers))

# Train SEPERATE Doc2Vec models for the mutant and interactor proteins
d2v_model1 = Doc2Vec(vector_size=dim1, min_count=1, alpha=alpha1, dm=dm1, window=w1)
d2v_model1.build_vocab(mut_train_corpus)
print('d2v 1 build vocab done')
d2v_model1.train(mut_train_corpus, total_examples=d2v_model1.corpus_count, epochs=epochs1) 
print('d2v 1 training done')

d2v_model2 = Doc2Vec(vector_size=dim2, min_count=1, alpha=alpha2, dm=dm2, window=w2)
d2v_model2.build_vocab(int_train_corpus)
print('d2v 2 build vocab done')
d2v_model2.train(int_train_corpus, total_examples=d2v_model2.corpus_count, epochs=epochs2) 
print('d2v 2 training done')

# SUM the mutant and interactor embeddings together to represent the PPI
mut_train_features = d2v_model1.dv
int_train_features = d2v_model2.dv
train_features = []
for i in range(len(int_train_features)):
    train_features.append(np.sum([int_train_features[i],mut_train_features[i]], axis=0)) # sum so dim is still 128
print(train_features[0].shape) # 128 (dim)!

# Save the summed embeddings
all_vecs = train_features
df['Vectors'] = all_vecs.tolist()

# Run SCV 
scv_df = pd.DataFrame(columns=['AUCs','PermYAUCs','F1s','Precisions','Recalls','TPRs','FPRs','FPRperms','TPRperms'])

# Number of iterations
outer_loop = 10
inner_loop = 10

all_tts_aucs = []
all_tts_permy_aucs = []
all_tts_f1s = []
all_tts_percisions =[]
all_tts_recalls = []
all_tts_fprs = []
all_tts_tprs = []
all_tts_permy_fprs = []
all_tts_permy_tprs = []

#  Make unshuffled dataset
features = np.array(list(df.Vectors.values))
labels = np.array(list(df.Y2H_score.values))

for o_its in tqdm(range(outer_loop)): 
    # Shuffle dataset to eliminate batch effect, change seed each time
    np.random.seed(o_its)
    
    # Create new shuffled indicies
    indices = np.arange(features.shape[0]) 
    np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]

    # Save the scores from AUCs
    tts_aucs = []   
    tts_permy_aucs = []
    tts_f1s = []
    tts_percisions =[]
    tts_recalls = []
    tts_fprs = []
    tts_tprs = []
    tts_permy_fprs = []
    tts_permy_tprs = []
    
    for i_its in range(inner_loop): # Shuffle each TTS
        # TTS + Permuted Y
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=i_its) # change seed
        y_test_perm = np.random.binomial(n=1, p=(Counter(y_test)[1.0]/len(y_test)), size=[len(y_test)]) # random ys

        # Train XGB
        xgb_cl = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
        xgb_cl.fit(X_train,y_train)

        # Test XGB on X_test and Permuted Y
        test_pred = xgb_cl.predict(X_test)
        pred_proba = xgb_cl.predict_proba(X_test)[:,1]
        fpr, tpr, _ = metrics.roc_curve(y_test, pred_proba)
        tts_fprs.append(fpr)
        tts_tprs.append(tpr)

        test_pred_perm = xgb_cl.predict(X_test)
        pred_proba_perm = xgb_cl.predict_proba(X_test)[:,1]
        fpr_perm, tpr_perm, _ = metrics.roc_curve(y_test_perm, pred_proba_perm)
        tts_permy_fprs.append(fpr_perm)
        tts_permy_tprs.append(tpr_perm)

        auc_score = metrics.roc_auc_score(y_test,pred_proba)
        auc_score_perm = metrics.roc_auc_score(y_test_perm,pred_proba_perm)
        f1_score =  metrics.f1_score(y_test,test_pred)
        precision = metrics.precision_score(y_test,test_pred)
        recall = metrics.recall_score(y_test,test_pred)
        
        tts_aucs.append(auc_score)
        tts_permy_aucs.append(auc_score_perm)
        tts_f1s.append(f1_score)
        tts_percisions.append(precision)
        tts_recalls.append(recall)
        
    # Define a common set of FPR values for plotting
    common_fpr = np.linspace(0, 1, 100)
    common_fpr_perm = np.linspace(0, 1, 100)
    interp_tpr = []
    interp_tpr_perm = []
    
    # Interpolate TPR values for each fold
    for i in range(len(tts_fprs)):
        interp_tpr.append(np.interp(common_fpr, tts_fprs[i], tts_tprs[i]))
        interp_tpr_perm.append(np.interp(common_fpr_perm, tts_permy_fprs[i], tts_permy_tprs[i]))
    # Calculate the mean of the interpolated TPR values
    mean_tpr = np.mean(interp_tpr, axis=0)
    mean_tpr_perm = np.mean(interp_tpr_perm, axis=0)

    all_tts_aucs.append(np.mean(tts_aucs))
    all_tts_permy_aucs.append(np.mean(tts_permy_aucs))
    all_tts_f1s.append(np.mean(tts_f1s))
    all_tts_percisions.append(np.mean(tts_percisions))
    all_tts_recalls.append(np.mean(tts_recalls))
    all_tts_fprs.append(common_fpr)
    all_tts_tprs.append(mean_tpr)
    all_tts_permy_fprs.append(common_fpr_perm)
    all_tts_permy_tprs.append(mean_tpr_perm)
    print('Outer Loop #' + str(o_its)+ ' Done')

# Save results
scv_df['AUCs'] = all_tts_aucs
scv_df['PermYAUCs'] = all_tts_permy_aucs
scv_df['F1s'] = all_tts_f1s
scv_df['Precisions'] = all_tts_percisions
scv_df['Recalls'] = all_tts_recalls
scv_df['FPRs'] = all_tts_fprs
scv_df['TPRs'] = all_tts_tprs
scv_df['FPRperms'] = all_tts_permy_fprs
scv_df['TPRperms'] = all_tts_permy_tprs
scv_df.to_pickle('EmbedSum_MutInt_SCV_example_results.pkl')
