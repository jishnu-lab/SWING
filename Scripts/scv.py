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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import argparse
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser('Nested Cross Validation of  MHC:Peptide Binding.')
parser.add_argument('--data_set',required=True,help='data to train with',default="data.csv")
parser.add_argument('--output',required=True,help='output prefix',default='1')
parser.add_argument('--loops',default=10,type=int,help='times for the loop')
#Language parameters:
parser.add_argument('--k',default=7,type=int,help='size of words (kmer) for the window encoding to be fragmented into')
parser.add_argument('--metric',required=False,help='The biochemical metric for calculating the scores between AA. Options: ["polarity","hydrophobic"]',default="polarity")
parser.add_argument('--padding_score',default=9,type=int,help='The value for padding. Should be outside the range of delta biochemical metric score.')
#Doc2Vec parameters
parser.add_argument('--w',default=7,type=int,help='how large the surrounding window is')
parser.add_argument('--dm',default=0,type=int,help='fill dm info')
parser.add_argument('--dim',default=72,type=int,help='fill dim')
parser.add_argument('--epochs',default=16,type=int,help='number epochs')
parser.add_argument('--min_count',default=1,type=int,help='fill minimum count')
parser.add_argument('--alpha',default=0.42,type=float,help='alpha value')
parser.add_argument('--save_embeddings',default=True, type=bool, help='Option to save Doc2Vec embeddings, to run cross-prediction with other seeds.')
#Classifier
parser.add_argument('--classifier',required=False,help='Specify the classifier to be used. Options: ["XGBoost","LR"]',default="XGBoost")
#XGBoost parameters
parser.add_argument('--n_estimators',default=200,type=int,help='fill n estimators')
parser.add_argument('--max_depth',default=1,type=int,help='maximum dempth')
parser.add_argument('--learning_rate',default=0.71,type=float,help='learning rate')
#Logistic Regression Parameters
parser.add_argument('--max_iter',default=10000,type=int,help='LR maximum iterations')
parser.add_argument('--l1_ratio',default=0.5,type=float,help='LR l1 ratio')

args = parser.parse_args()

#creating output directory

os.makedirs("output", exist_ok=True)
os.makedirs("output/scv", exist_ok=True)
os.makedirs("output/models/", exist_ok=True)

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

def get_window_encodings(df, padding_score=9): # takes df (epitope/receptor sequences) and window (size of epitope).
    """
    Takes a pandas dataframe where each row represents a protein-protein/peptide-protein interaction.  
    Customization includes setting the interactor protein and the peptide window. In the pMHC context, the epitope defines the peptide window. In the missense mutation pertubation context, the window_k parameter defines the size of the window and the mutation defines the position. Additionally, the scale used to calculate the score can be altered. If the scale is changed the padding_score may need to be adjusted.  
    The function returns a list of score encodings strings that each represent a PPI. The ends of the encodings include padding from the sliding window process. These encodings will be broken into k-mers for the embedding model.
    """
    total_encodings = [] # Master list of encodings
    for i in (df.index): # Iterate through protein pairs
        mut_window = df['Epitope'].iloc[i] # Epitope to slide
        interactor = df['Sequence'].iloc[i] # Interacting sequence
        PPI_encoding = '' # For each PPI
        its = 0 # Tracks sliding window position
        for j in range(len(interactor)): # For the entire length of the interactor
            window_scores = ''  # Saves the scores between window-interactor at the 'its' position
            for k in range(len(mut_window)): # At each positon of the interactor ('its'), align epitope and find the score differences
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

    return total_encodings

def get_kmers_str(encoding_scores,k=7,shuffle=False, padding_score=9):
    """
    Takes the encoding scores from get_window_encodings().  
    Customization includes setting size of the kmers (k), a shuffle option, and the integer defining the padding score.  
    This function returns a list of lists of overlapping k-mers of specified size k, removing k-mers of only padding. Each list of k-mers are specific to each of the PPIs. This output is compatible with gensims
    """
    padding = {str(padding_score)}
    for i in range(k): # Makes a set of padding scores that will be removed from the final k-mers 
        padding.add(str(padding_score)*(i+1)) # {'9','99','999'...}
    kmers = [] # Master list of k-mers
    for ppi_score in encoding_scores: # For each PPI encoding
        int_kmers = [] # K-mers specific to PPI
        for j in range(len(ppi_score)-k): # Iterate over the PPI encoding
            kmer = ppi_score[j:j+k] # Slice k-mers and sliding over
            if kmer not in padding: # If K-mer is just padding, don't add it
                int_kmers.append(kmer) # Keep non-padding k-mers  
        if shuffle: # shuffle option
            random.shuffle(int_kmers)
        kmers.append(int_kmers) # Append k-mers to master list
    return kmers

def get_corpus(matrix, tokens_only=False): # turns each ppi into a d2v tagged document
    """
    Takes in the k-mers created by the get_kmers_str() function.  
    Returns a Doc2Vec TaggedDocuments entities for each PPI to be used in a Doc2Vec model.
    """
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

df = pd.read_csv(args.data_set)
data = df.sample(frac = 1).reset_index(drop=True)

inner_loop = args.loops

all_tts_aucs = []
all_tts_permy_aucs = []
all_tts_f1s = []
all_tts_precisions =[]
all_tts_recalls = []
all_tts_fprs = []
all_tts_tprs = []
all_best_thresh = []
all_tts_permy_fprs = []
all_tts_permy_tprs = []
all_avg_precisions = []

window_encodings = get_window_encodings(data, padding_score=args.padding_score) # still encode all and get k-mers
kmers = get_kmers_str(window_encodings,k=k, padding_score=args.padding_score)
train_corpus = list(get_corpus(kmers))

d2v_model = Doc2Vec(vector_size=dim,dm=dm,window=w,min_count=min_count,alpha=alpha)
d2v_model.build_vocab(train_corpus) # d2v trains on everything
d2v_model.train(train_corpus,total_examples=d2v_model.corpus_count, epochs=epochs)

all_vecs = d2v_model.dv.vectors
data["Vectors"] = all_vecs.tolist()
 
if args.save_embeddings==True:
        print("Saving Doc2Vec model...")
        d2v_model.save("output/models/scv_doc2vec_{0}.model".format(output))
        data.to_csv('output/models/scv_{0}_shuffled_df_vectors.csv'.format(output), index=False)
else:
        print("Doc2Vec model not saved.")    

if args.classifier == 'XGBoost':
    print('Using XGBoost Classifier')

    #set parameters
    n_estimators = args.n_estimators
    max_depth = args.max_depth
    learning_rate = args.learning_rate
    
if args.classifier == 'LR':
    print('Using Logistic Regression Classifier')

    #set parameters
    max_iter = args.max_iter
    l1_ratio = args.l1_ratio

for i_its in range(inner_loop): # shuffle each tts of the 'data' set
    print("Outer Loop #{0}".format(i_its))
    #Shuffle the dataset
    data_shuffled = data.sample(frac = 1, random_state = i_its).reset_index(drop=True)
    data_shuffled_vectors = np.array(list(data_shuffled["Vectors"]))
    hits = np.array(data_shuffled['Hit'])
    
    tts_aucs = []
    tts_permy_aucs = []
    tts_f1s = []
    tts_precisions = []
    tts_recalls = []
    tts_fprs = []
    tts_tprs = []
    tts_threshs = []
    tts_best_thresh = []
    tts_permy_fprs = []
    tts_permy_tprs = []
    tts_average_precision = []    

    #Stratify the shuffled set
    skf = StratifiedKFold(n_splits=args.loops, shuffle=True, random_state=1)
    split_set = skf.split(data_shuffled['Vectors'], data_shuffled['Hit'])

    for i, (train_index, test_index) in enumerate(split_set):
        print("Inner Loop #{0}".format(i))
        X_test = []
        X_train = []
        y_test = []
        y_train = []
        for j in range(len(data_shuffled)):
            if j in train_index:
                X_train.append(data_shuffled_vectors[j])
                y_train.append(hits[j])
            else:
                X_test.append(data_shuffled_vectors[j])
                y_test.append(hits[j])

        #Random generation of Ys for permutation with same distribution as dataset.
        y_test_perm = np.random.binomial(n=1, p=0.1, size=[len(y_test)]) # random ys 

        if args.classifier == 'XGBoost':
             print("Setting XGBoost classifier_model")
             classifier_model = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)

        if args.classifier == 'LR':
             print("Setting LR classifier_model")
             classifier_model = LogisticRegression(random_state=0, solver = 'saga', penalty = 'elasticnet', max_iter=max_iter, l1_ratio=l1_ratio)

        classifier_model.fit(X_train,y_train)

    	# ROC curves
        test_pred = classifier_model.predict(X_test)
        pred_proba = classifier_model.predict_proba(X_test)[:,1]
        fpr, tpr, tts_thresh = metrics.roc_curve(y_test,  pred_proba)
        tts_fprs.append(fpr)
        tts_tprs.append(tpr)
    
    	# calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))

        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (tts_thresh[ix], gmeans[ix]))
        tts_best_thresh.append(tts_thresh[ix])

    	#permutation testing
        test_pred_perm = classifier_model.predict(X_test)
        pred_proba_perm = classifier_model.predict_proba(X_test)[:,1]
        fpr_perm, tpr_perm, _ = metrics.roc_curve(y_test_perm, pred_proba)
        tts_permy_fprs.append(fpr_perm)
        tts_permy_tprs.append(tpr_perm)

    	#record metrics
        auc_score = metrics.roc_auc_score(y_test,pred_proba)
        auc_score_perm = metrics.roc_auc_score(y_test_perm,pred_proba)
        f1_score =  metrics.f1_score(y_test,test_pred)
        precision = metrics.precision_score(y_test,test_pred)
        recall = metrics.recall_score(y_test,test_pred)
        avg_precision = metrics.average_precision_score(y_test,test_pred)

        tts_aucs.append(auc_score)
        tts_permy_aucs.append(auc_score_perm)
        tts_f1s.append(f1_score)
        tts_precisions.append(precision)
        tts_recalls.append(recall)
        tts_average_precision.append(avg_precision)
     
    # Define a common set of FPR values
    common_fpr = np.linspace(0, 1, 100)
    common_fpr_perm = np.linspace(0, 1, 100)
    interp_tpr = []
    interp_tpr_perm = []
    
    # Interpolate TPR values for each fold
    for i in range(len(tts_fprs)):
        interp_tpr.append(np.interp(common_fpr, tts_fprs[i], tts_tprs[i]))
        interp_tpr_perm.append(np.interp(common_fpr_perm, tts_permy_fprs[i], tts_permy_tprs[i]))
    
    #Calculate the mean of the interpolated TPR values
    mean_tpr = np.mean(interp_tpr, axis=0)
    mean_tpr_perm = np.mean(interp_tpr_perm, axis=0)
    
    all_tts_aucs.append(np.mean(tts_aucs))
    all_tts_permy_aucs.append(np.mean(tts_permy_aucs))
    all_tts_f1s.append(np.mean(tts_f1s))
    all_tts_precisions.append(np.mean(tts_precisions))
    all_tts_recalls.append(np.mean(tts_recalls))
    all_tts_fprs.append(common_fpr)
    all_tts_tprs.append(mean_tpr)
    all_tts_permy_fprs.append(common_fpr_perm)
    all_tts_permy_tprs.append(mean_tpr_perm)
    all_avg_precisions.append(np.mean(tts_average_precision))
    all_best_thresh.append(np.mean(tts_best_thresh))    
    print('Outer Loop #' + str(i_its)+ ' Done')

    np.save('output/scv/all_tts_aucs_{0}'.format(output), all_tts_aucs)
    np.save('output/scv/all_permy_aucs_{0}'.format(output), all_tts_permy_aucs)
    np.save('output/scv/all_fpr_tts_{0}'.format(output), all_tts_fprs)
    np.save('output/scv/all_tpr_tts_{0}'.format(output), all_tts_tprs)
    np.save('output/scv/all_best_thresh_{0}'.format(output), all_best_thresh)
    np.save('output/scv/all_tpr_perm_aucs_{0}'.format(output), all_tts_permy_tprs)
    np.save('output/scv/all_fpr_perm_aucs_{0}'.format(output), all_tts_permy_fprs)
    np.save('output/scv/all_f1_scores_{0}'.format(output), all_tts_f1s)
    np.save('output/scv/all_recalls_{0}'.format(output), all_tts_recalls)
    np.save('output/scv/all_precisions_{0}'.format(output), all_tts_precisions)
    np.save('output/scv/all_avg_precisions_{0}'.format(output), all_avg_precisions)
    
