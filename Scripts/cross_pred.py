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
from sklearn.metrics import average_precision_score
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser('Cross Prediction of  MHC:Peptide Binding.')
parser.add_argument('--data_set',required=True,help='Dataset with "Set" column labeling Train and Test sets',default="data.csv")
parser.add_argument('--output',required=True,help='output prefix for saving files',default='1')
parser.add_argument('--cross_pred_set',required=True,help='The prefix for the cross-pred dataset for saving files and labeling graphs',default='HLA-A02:02')
parser.add_argument('--loops',default=10,type=int,help='How many stratified k-folds')
#Language Parameters:
parser.add_argument('--metric',required=False,help='The biochemical metric for calculating the scores between AA. Options: ["polarity","hydrophobicity"]',default="polarity")
parser.add_argument('--k',default=7,type=int,help='size of words (kmer) for the window encoding to be fragmented into')
parser.add_argument('--padding_score',default=9,type=int,help='The value for padding. Should be outside the range of delta biochemical metric score.')
#Doc2Vec Parameters
parser.add_argument('--w',default=7,type=int,help='how large the surrounding window is')
parser.add_argument('--dm',default=0,type=int,help='Doc2Vec dm parameter')
parser.add_argument('--dim',default=72,type=int,help='Doc2Vec dim parameter')
parser.add_argument('--epochs',default=16,type=int,help='Doc2Vec number epochs')
parser.add_argument('--min_count',default=1,type=int,help='Doc2Vec minimum count')
parser.add_argument('--alpha',default=0.42,type=float,help='Doc2Vec alpha value')
parser.add_argument('--save_embeddings',default=True, type=bool, help='Option to save Doc2Vec embeddings, to run cross-prediction with other seeds.')
#Classifier
parser.add_argument('--classifier',required=False,help='Specify the classifier to be used. Options: ["XGBoost","LR"]',default="XGBoost")
#XGBoost Parameters
parser.add_argument('--n_estimators',default=200,type=int,help='XGBoost n estimators')
parser.add_argument('--max_depth',default=1,type=int,help='XGBoost maximum depth')
parser.add_argument('--learning_rate',default=0.71,type=float,help='XGBoost learning rate')
#Logistic Regression Parameters
parser.add_argument('--max_iter',default=10000,type=int,help='LR maximum iterations')
parser.add_argument('--l1_ratio',default=0.5,type=float,help='LR l1 ratio')

args = parser.parse_args()

#creating output directory
os.makedirs("output", exist_ok=True)
os.makedirs("output/cross_pred", exist_ok=True)
os.makedirs("output/cross_pred/dataframes", exist_ok=True)
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
    for i in range(len(matrix)):
        yield gensim.models.doc2vec.TaggedDocument(matrix[i],[i])

k = args.k
dim = args.dim
dm = args.dm
w = args.w
min_count = args.min_count
alpha = args.alpha
epochs = args.epochs
loops = args.loops
output = args.output
cross_pred_set = args.cross_pred_set
loops = args.loops
classifier = args.classifier
padding_score = args.padding_score

print('Model:', args.data_set, 'cross predicting on: ',  cross_pred_set)

df = pd.read_csv(args.data_set)
df = df.sample(frac = 1, random_state = 1).reset_index() #shuffle the dataset.
#df = df.drop(columns = ['index','Unnamed: 0'])

window_encodings = get_window_encodings(df, padding_score=padding_score)
initial_kmers = get_kmers_str(window_encodings,k=k,shuffle=False, padding_score=padding_score) # no shuffle

kmers = []

for i in range(len(initial_kmers)):
    if i in set(df[df['Set']=='Test'].index):
        kmers.append(initial_kmers[i])
        #print(initial_kmers[i])

    else:
        random.shuffle(initial_kmers[i])
        kmers.append(initial_kmers[i]) # shuffle just the train set    

#Embed the whole dataset:        
train_corpus = list(get_corpus(kmers))

d2v_model = Doc2Vec(vector_size=dim,dm=dm,window=w,min_count=min_count,alpha=alpha)
d2v_model.build_vocab(train_corpus)
d2v_model.train(train_corpus,total_examples=d2v_model.corpus_count, epochs=epochs)

all_vecs = d2v_model.dv.vectors
df["Vectors"] = all_vecs.tolist()

if args.save_embeddings==True:
        print("Saving Doc2Vec model...")
        d2v_model.save("output/models/doc2vec_{0}.model".format(output))

#all_vecs = d2v_model.dv.vectors
        df.to_csv('output/models/{0}_shuffled_df_vectors.csv'.format(output), index=False)

all_fpr = []
all_tpr = []
all_threshs = []
all_rocs = []
all_best_threshs = []
all_precisions = []
all_recalls = []
all_pr_thresh = []
all_f1_scores = []
all_avg_precisions = []

#Split the dataframe into the test and train sets
test_df = df[df['Set']=='Test']
train_df = df[df['Set']!='Test']

test_vectors = np.array(list(test_df["Vectors"]))
train_vectors = np.array(list(train_df["Vectors"]))
test_epitopes = np.array(test_df['Epitope'])
test_mhcs = np.array(test_df['MHC'])
#test_seq = np.array(test_df['Sequence'])
test_hit = np.array(test_df['Hit'])

#Stratify the Test set only
skf = StratifiedKFold(n_splits=loops, shuffle=True, random_state=1)
split_set = skf.split(test_df['Vectors'], test_df['Hit'])

#Set the training vectors
tr_vecs = []

for l in range(len(train_vectors)):
    tr_vecs.append(train_vectors[l])

if classifier == 'XGBoost':
    print('Using XGBoost Classifier')

    #set parameters
    n_estimators = args.n_estimators
    max_depth = args.max_depth
    learning_rate = args.learning_rate
        
    #Get training set Y labels
    tr_y = train_df.Hit.values.reshape(-1,1)

    # train xbg on all the training Data
    classifier_model = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)

if args.classifier == 'LR':
        print('Using Logistic Regression Classifier')

        #set parameters
        max_iter = args.max_iter
        l1_ratio = args.l1_ratio

        #Split labels (Y) into Train and Test sets
        tr_y = np.ravel(train_df.Hit.values.reshape(-1,1))

        #train LR on all the training data
        classifier_model = LogisticRegression(random_state=0, solver = 'saga', penalty = 'elasticnet', max_iter=max_iter, l1_ratio=l1_ratio)

classifier_model.fit(tr_vecs,tr_y)

for i, (test_index, other_index) in enumerate(split_set):
    print('Cross Prediction #{0}'.format(i))

    #Split vectors (X) into train and test sets:
    print("Spliting test vectors")
    
    test_vecs = []
    test_y = []
    epitopes_list = []
    mhc_list = []
    
    for j in range(len(test_df)):
        if j in test_index:
            test_vecs.append(test_vectors[j])
            test_y.append(test_hit[j])
            epitopes_list.append(test_epitopes[j])
            mhc_list.append(test_mhcs[j])

    print('Making Predictions...')
    #classifier predictions:
    yval_pred = classifier_model.predict(test_vecs)
    yval_pred_proba = classifier_model.predict_proba(test_vecs)[::,1]
    fpr, tpr, thresh = metrics.roc_curve(test_y,  yval_pred_proba)
    roc_auc = auc(fpr,tpr)
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))

    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

    #measure precesion recall
    precision, recall, pr_thresh = precision_recall_curve(test_y,  yval_pred_proba)

    #measure f1_score
    f1_score = metrics.f1_score(test_y, yval_pred)

    #measure average precision 
    average_precision = average_precision_score(test_y, yval_pred_proba)
    
    all_fpr.append(list(fpr))
    all_tpr.append(list(tpr))
    all_threshs.append(list(thresh))
    all_rocs.append(roc_auc)
    all_best_threshs.append(thresh[ix])
    all_precisions.append(list(precision))
    all_recalls.append(list(recall))
    all_f1_scores.append(f1_score)
    all_avg_precisions.append(average_precision)
    all_pr_thresh.append(list(pr_thresh))

    print('AUC ROC Set score {0} Data: '.format(cross_pred_set) + str(roc_auc))

    np.save('output/cross_pred/rocauc_{0}'.format(output), all_rocs)
    np.save('output/cross_pred/fpr_{0}'.format(output), all_fpr)
    np.save('output/cross_pred/tpr_{0}'.format(output), all_tpr)
    np.save('output/cross_pred/thresh_{0}'.format(output), all_threshs)
    np.save('output/cross_pred/best_thresh_{0}'.format(output), all_best_threshs)
    np.save('output/cross_pred/precision_{0}'.format(output), all_precisions)
    np.save('output/cross_pred/recall_{0}'.format(output), all_recalls)
    np.save('output/cross_pred/prthresh_{0}'.format(output), pr_thresh)
    np.save('output/cross_pred/f1_score_{0}'.format(output), all_f1_scores)
    np.save('output/cross_pred/average_precision_{0}'.format(output), all_avg_precisions)

    #create output dataframe with the probabilities and predictions
    df_prediction = pd.DataFrame()
    df_prediction['Epitope'] = epitopes_list
    df_prediction['MHC'] = mhc_list
    df_prediction['True Y'] = test_y
    df_prediction['Predicted Y'] =  yval_pred
    df_prediction['Predicted Probabilities Y'] = yval_pred_proba
    df_prediction = df_prediction.sort_values(by='Predicted Probabilities Y', ascending=False)
    df_prediction.to_csv('output/cross_pred/dataframes/predictions_{0}_{1}.csv'.format(output, i), index=False)
