import pandas as pd 
from xgboost import XGBClassifier 
import numpy as np
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import f1_score, roc_auc_score , roc_curve, average_precision_score, precision_recall_curve 
from sklearn.linear_model import LogisticRegression
import sys
import argparse 
import os 
#import wandb 
# commandline args for the different inputs to the model

parser = argparse.ArgumentParser()

parser.add_argument("--n_estimators",type=int,help="The number of trees for the XGBoost model")
parser.add_argument("--max_depth",type=int,help="The maximum depth of the trees")
parser.add_argument("--learning_rate",type=float,help="The learning rate")
parser.add_argument("--run_name",type=str,help="name of model training run")
parser.add_argument("--project_name",type=str,help="wandb project name")

args = parser.parse_args()

#wandb.init(project = args.project_name, name = args.run_name, config = args)

data = pd.read_csv("npflip_nature_mut_wt_merged_vocab_k7_subsize7.csv",sep=",",header=0)

encoded_lang = list(data["Encoded"])

lens = [len(item) for item in encoded_lang]

max_len = max(lens)

print(max_len)

#sys.exit()

features = []

for seq in encoded_lang:
	seq = str(seq)
	tmp= []
	for i in range(max_len):
		try:
			tmp.append(int(seq[i]))
		except IndexError:
			tmp.append(-1)

	features.append(tmp)


y = list(data["Y2H_score"])

print(len(features))
print(len(features[0]))

x = np.array(features)
y = np.array(y)

skf = StratifiedKFold(n_splits=5)	

aucs = []
f1s = []

model = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate)

#model = LogisticRegression(C=10,max_iter=1000)


features = x 
labels = y

all_fpr = []
all_tpr = []
all_rocs =[]
all_precisions = []
all_avg_precisions = []
all_f1_scores = []
outer_loop =5


tmp = np.column_stack((features,labels))
for j in range(outer_loop):
        # shuffle the order of rows for tmp
        np.random.shuffle(tmp)
        features = tmp[:,:-1]
        labels = tmp[:,-1]
        tts_aucs = []
        tts_f1s = []
        tts_precisions = []
        tts_recalls = []
        tts_fprs = []
        tts_tprs = []
        tts_average_precision = []

        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)

        #print(features)
        #print(labels)

        for i, (train_index,test_index) in enumerate(skf.split(features,labels)):

                X_tr, y_tr = features[train_index],labels[train_index]

                X_te, y_te =  features[test_index], labels[test_index]


                #print(X_tr,y_tr)

                print(type(X_tr))

                print(type(y_tr))
                model.fit(X_tr,y_tr)

                pred_scores = model.predict_proba(X_te)[:,1]

                auc = roc_auc_score(y_te,pred_scores)

                fpr,tpr,thresh = roc_curve(y_te,pred_scores)

                tts_tprs.append(tpr)
                tts_fprs.append(fpr)
                tts_aucs.append(auc)

                prec,recall,thresh = precision_recall_curve(y_te,pred_scores)

                preds = model.predict(X_te)

                f1 = f1_score(y_te,preds)

                average_precision = average_precision_score(y_te,pred_scores)

                tts_precisions.append(prec)
                tts_recalls.append(recall)
                tts_f1s.append(f1)
                tts_average_precision.append(average_precision)

                # define a common set of fpr values
                common_fpr = np.linspace(0,1,100)
                interp_tpr = []
                for i in range(len(tts_fprs)):
                        interp_tpr.append(np.interp(common_fpr,tts_fprs[i],tts_tprs[i]))


                mean_tpr = np.mean(interp_tpr,axis=0)

                tts_precisions1 = np.array(tts_precisions,dtype=object)
                tts_recalls1 = np.array(tts_recalls,dtype=object)
                tts_f1s1 = np.array(tts_f1s,dtype=object)
                tts_average_precision1 = np.array(tts_average_precision,dtype=object)

                all_fpr.append(common_fpr)
                all_tpr.append(mean_tpr)
                all_rocs.append(np.mean(tts_aucs))
                #all_precisions.append(np.mean(tts_precisions1))
                #all_avg_precisions.append(np.mean(tts_average_precision1))

                all_f1_scores.append(np.mean(tts_f1s))


path = "resultsOnlyILM/"

name = "SCV"

roc_fpr = os.path.join(path,name+".mutint.onlyILM.fpr.npy")
roc_tpr = os.path.join(path,name+".mutint.onlyILM.tpr.npy")
pr_prec = os.path.join(path,name+".mutint.onlyILM.precision.npy")
pr_recall = os.path.join(path,name+".mutint.onlyILM.recall.npy")
roc_auc = os.path.join(path,name+".mutint.onlyILM.auroc.npy")
f1_scores = os.path.join(path,name+".mutint.onlyILM.f1.npy")

np.save(open(roc_fpr,'wb'),np.array(all_fpr,dtype=object))
np.save(open(roc_tpr,"wb"),np.array(all_tpr,dtype=object))

#np.save(open(pr_prec,'wb'),np.array(all_precisions,dtype=object))
#np.save(open(pr_recall,'wb'),np.array(all_recalls,dtype=object))

np.save(open(roc_auc,'wb'),np.array(all_rocs,dtype=object))
np.save(open(f1_scores,'wb'),np.array(all_f1_scores,dtype=object))
