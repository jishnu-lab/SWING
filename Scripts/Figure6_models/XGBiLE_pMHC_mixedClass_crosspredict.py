import pandas as pd
from xgboost import XGBClassifier 
import numpy as np 
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, roc_curve , auc, average_precision_score 
from sklearn.model_selection import StratifiedKFold, train_test_split 
import os 

def encode(df):
    encoded_lang = list(df["Encoded"])
    lens = [len(item) for item in encoded_lang]
    max_len = 4410
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
    y = list(df["Hit"])
    x = np.array(features)
    y = np.array(y)

    return x,y 

def eval(model,x,y):
    scores = model.predict_proba(x)[:,1]
    auc = roc_auc_score(y,scores)
    preds = model.predict(x)
    f1 = f1_score(y,preds)

    return auc,f1

path = "forFigures/"

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)

def curves(model,x,y,name):
    split_set = skf.split(x,y)
    all_fpr = []
    all_tpr = []
    all_rocs =[]
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_avg_precisions = []

    for i,(test_index,_) in enumerate(split_set):
        x1,y1 = x[test_index], y[test_index]


        scores = model.predict_proba(x1)[:,1]
        preds = model.predict(x1)
        fpr,tpr,thresh = roc_curve(y1,scores)
        prec,recall,thresh1 = precision_recall_curve(y1,scores)
        roc_auc = roc_auc_score(y1,scores)
        f1 = f1_score(y1,preds)
        average_precision = average_precision_score(y1,scores)

        all_fpr.append(list(fpr))
        all_tpr.append(list(tpr))
        all_rocs.append(roc_auc)
        all_precisions.append(list(prec))
        all_recalls.append(list(recall))
        all_f1_scores.append(f1)
        #all_pr_thresh.append(list(pr_thresh))

    roc_fpr = os.path.join(path,name+".mixed.fpr.npy")
    roc_tpr = os.path.join(path,name+".mixed.tpr.npy")
    pr_prec = os.path.join(path,name+".mixed.precision.npy")
    pr_recall = os.path.join(path,name+".mixed.recall.npy")
    roc_auc = os.path.join(path,name+".mixed.auroc.npy")
    f1_scores = os.path.join(path,name+".mixed.f1.npy")
    np.save(open(roc_fpr,'wb'),all_fpr)
    np.save(open(roc_tpr,"wb"),all_tpr)
    np.save(open(pr_prec,'wb'),all_precisions)
    np.save(open(pr_recall,'wb'),all_recalls)
    np.save(open(roc_auc,'wb'),all_rocs)
    np.save(open(f1_scores,'wb'),all_f1_scores)

data = pd.read_csv("MixedClass_training_210_vocab_k7_subsize7.csv",sep=",",header=0)

x,y = encode(data)


model = XGBClassifier(n_estimators=66,max_depth=7,learning_rate=0.23707)

model.fit(x,y)


# cross predictions 

data_a02 = pd.read_csv("hla.A02:02.mixedVal.swing.encoded.training210.csv",sep=",",header=0)

x_a,y_a = encode(data_a02)

auc, f1 = eval(model,x_a,y_a)

curves(model,x_a,y_a,"HLA-A02:02")

print("The scores for A02:02",auc,f1)

data_b40 = pd.read_csv("hla.B40:02.mixedVal.swing.encoded.training210.csv",sep=",",header=0)

x_b,y_b = encode(data_b40)

auc,f1 = eval(model,x_b,y_b)

curves(model,x_b,y_b,"HLA-B40:02")

print("the scores for B40:",auc,f1)

data_c05 = pd.read_csv("hla.C05:01.mixedVal.swing.encoded.training210.csv",sep=",",header=0)

x_b,y_b = encode(data_c05)

auc,f1 = eval(model,x_b,y_b)

curves(model,x_b,y_b,"HLA-C05:01")
print("the scores for C05: ", auc,f1)

data_drb1 = pd.read_csv("hla.drb1:0102.mixedVal.swing.encoded.training210.csv",sep=",",header=0)

x_b,y_b = encode(data_drb1)

auc,f1 = eval(model,x_b,y_b)

curves(model,x_b,y_b,"HLA-DRB1:0102")

print("the scores for drb1:0102:",auc,f1)

data_drb04 = pd.read_csv("hla.drb1:0404.mixedVal.swing.encoded.training210.csv",sep=",",header=0)

x_b,y_b = encode(data_drb04)

auc,f1 = eval(model,x_b,y_b)

curves(model,x_b,y_b,"HLA-DRB1:0404")

print("the scores for drb1:0404:",auc,f1)

data_drb04 = pd.read_csv("iag7.swing.encoded.training210.csv",sep=",",header=0)

x_b,y_b = encode(data_drb04)

auc,f1 = eval(model,x_b,y_b)

curves(model,x_b,y_b,"IAG7")


print("the scores for IAG7:",auc,f1)

data_drb04 = pd.read_csv("iek.swing.encoded.training210.csv",sep=",",header=0)

x_b,y_b = encode(data_drb04)

auc,f1 = eval(model,x_b,y_b)

curves(model,x_b,y_b,"IEk")

print("the scores for IEk:",auc,f1)
