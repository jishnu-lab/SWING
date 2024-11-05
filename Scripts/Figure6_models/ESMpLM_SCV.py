import pickle as pkl
import pandas as pd 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score , f1_score , roc_curve, precision_recall_curve
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier 
import numpy as np 
import torch
import os
import argparse 
import wandb 

# read in the dataset

parser = argparse.ArgumentParser()

parser.add_argument("--n_estimators",type=int,help="The number of trees for the XGBoost model")
parser.add_argument("--max_depth",type=int,help="The maximum depth of the trees")
parser.add_argument("--learning_rate",type=float,help="The learning rate")
parser.add_argument("--run_name",type=str,help="name of model training run")
parser.add_argument("--project_name",type=str,help="wandb project name")

args = parser.parse_args()

#wandb.init(project = args.project_name, name = args.run_name, config = args)

data = pd.read_csv("npflip_nature_mut_wt_merged.csv",sep=",",header=0)

info = list(zip(data["Target_UPID"],data["Mutation"],data["Interactor_UPID"],data["Type"],data["Y2H_score"]))


features = []
labels = []

path2embeds = "embeds/"

for target, mut, interactor, types, label in info:

        target_id = target+"_" + mut + "_" + types

        int_id = interactor + "_" + types 

        #target_index = ids2index[target_id]

        #int_index = ids2index[int_id]

        feat_target = torch.load(os.path.join(path2embeds,target_id+" <unknown description>.pt"))

        feat_target = np.array(feat_target["mean_representations"][36])

        feat_int = torch.load(os.path.join(path2embeds,int_id+" <unknown description>.pt"))
        feat_int = np.array(feat_int["mean_representations"][36])


        #feat_target = embeds[target_index,:]

        #feat_int = embeds[int_index,nteractor, types, label in info:

        #target_index = ids2index[target_id]

        #int_index = ids2index[int_id]

       
        feat = feat_target + feat_int 

        features.append(feat.tolist())

        labels.append(label)
	
	
#print(len(features[0]))

# init the classifier 

model = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate)

scores = []

skf = StratifiedKFold(n_splits=5)

features = np.array(features)
labels = np.array(labels)

f1s = []


all_fpr = []
all_tpr = []
all_rocs =[]
all_f1_scores = []
all_avg_precisions = []

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
                #all_recalls.append(np.mean(tts_recalls1))
                all_f1_scores.append(np.mean(tts_f1s))
                #all_avg_precisions.append(np.mean(tts_average_precision1))

path = "resultsESMplm/"

name = "SCV"

roc_fpr = os.path.join(path,name+".mutint.ESM1-PLM.fpr.npy")
roc_tpr = os.path.join(path,name+".mutint.ESM1-PLM.tpr.npy")
pr_prec = os.path.join(path,name+".mutint.ESM1-PLM.precision.npy")
pr_recall = os.path.join(path,name+".mutint.ESM1-PLM.recall.npy")
roc_auc = os.path.join(path,name+".mutint.ESM1-PLM.auroc.npy")
f1_scores = os.path.join(path,name+".mutint.ESM1-PLM.f1.npy")

np.save(open(roc_fpr,'wb'),np.array(all_fpr,dtype=object))
np.save(open(roc_tpr,"wb"),np.array(all_tpr,dtype=object))
#np.save(open(pr_prec,'wb'),np.array(all_precisions,dtype=object))
#np.save(open(pr_recall,'wb'),np.array(all_recalls,dtype=object))
np.save(open(roc_auc,'wb'),np.array(all_rocs,dtype=object))
np.save(open(f1_scores,'wb'),np.array(all_f1_scores,dtype=object))

print("Done saving!")


