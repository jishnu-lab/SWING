import torch 
import torch.nn as nn 
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve  
from sklearn.model_selection import StratifiedKFold
import os 
import torch.nn.functional as F 
import warnings 
from collections import OrderedDict
#from CNNClassI import ilmCNN 


class ilmCNN(nn.Module):
	def __init__(self,input_dim, embedding_dim,num_classes,num_kernels,kernel_size,dropout):
		super(ilmCNN,self).__init__()
		self.embedding = nn.Embedding(input_dim,embedding_dim)
		self.conv1 = nn.Conv1d(embedding_dim, num_kernels, kernel_size)
		self.fc1 = nn.Linear(num_kernels,128)
		self.fc2 = nn.Linear(128,num_classes)

		self.dropout = nn.Dropout(dropout)

	def forward(self,x):
		x = self.embedding(x)
		x = x.permute(0,2,1)
		x = F.relu(self.conv1(x)) # conv layer 1
	
		x = F.max_pool1d(x,x.size(2)).squeeze(2) # global max pooling
		x = self.dropout(F.relu(self.fc1(x))) # fully connected layer 2

		x = self.fc2(x) # Fully connected layer 2

		return x


warnings.filterwarnings("ignore")

# read in the model 

checkpoint  = torch.load("mixedClass.pmhc.210.trained.1dCNN.pth")

state_dict = checkpoint.state_dict()

#print(state_dict)

#print(checkpoint)

#new_state_dict = OrderedDict()

#del state_dict["embedding.weight"]


model = ilmCNN(11,27,2,118,5,0.48176)
model.load_state_dict(state_dict)

model.eval()

def sequences_to_indixes(df,max_length):
	encodings = list(df["Encoded"])
	all_indices = []
	for seq in encodings:
		indices = []
		for item in seq:
			indices.append(int(item))

		if len(indices) < max_length:
			indices += [10] * (max_length - len(indices)) # padding

		else:
			indices = indices[:max_length]
		all_indices.append(indices)
	return all_indices

path = "resultsCNN/"

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)


def eval(X,y,batch_size,name):

	split_set = skf.split(X,y)
	all_fpr = []
	all_tpr = []
	all_rocs = []
	all_precisions = []
	all_recalls = []
	all_f1_scores = []

	for i, (test_index,_) in enumerate(split_set):
		X1,y1 = X[test_index],y[test_index]
		

		model.eval()
		# evalute the model with different validation datasets
		test_dataset = torch.utils.data.TensorDataset(torch.tensor(X),torch.tensor(y))
		test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

		all_preds = []
		all_labels = []

		with torch.no_grad():
			for inputs,labels in test_loader:
				outputs = model(inputs)
				preds = F.softmax(outputs,dim=1)[:,1]
				all_preds.append(preds.cpu().numpy())
				all_labels.append(labels.cpu().numpy())

		all_preds = np.concatenate(all_preds)
		all_labels = np.concatenate(all_labels)
		pred_labels = []
		for item in all_preds.tolist():
			if item>0.5:
				pred_labels.append(1)
			else:
				pred_labels.append(0)

	
		auc = roc_auc_score(all_labels,all_preds)
		f1 = f1_score(all_labels,pred_labels)
		fpr,tpr,thresh = roc_curve(all_labels,all_preds)
		prec,recall,thresh1 = precision_recall_curve(all_labels,all_preds) 
		
		all_fpr.append(list(fpr)) 
		all_tpr.append(list(tpr))
		all_rocs.append(auc)
		all_precisions.append(list(prec)) 
		all_recalls.append(list(recall))
		all_f1_scores.append(f1) 

	roc_fpr = os.path.join(path,name+".mixed.fpr.npy")
	roc_tpr = os.path.join(path,name+".mixed.tpr.npy")
	pr_prec = os.path.join(path,name+".mixed.precision.npy")
	pr_recall = os.path.join(path,name+".mixed.recall.npy")
	roc_auc = os.path.join(path,name+".mixed.auroc.npy")
	f1_scores = os.path.join(path,name+".mixed.f1.npy")

	np.save(open(roc_fpr,'wb'),np.array(all_fpr,dtype=object))
	np.save(open(roc_tpr,"wb"),np.array(all_tpr,dtype=object))
	np.save(open(pr_prec,'wb'),np.array(all_precisions,dtype=object))
	np.save(open(pr_recall,'wb'),np.array(all_recalls,dtype=object))
	np.save(open(roc_auc,'wb'),np.array(all_rocs,dtype=object))
	np.save(open(f1_scores,'wb'),np.array(all_f1_scores,dtype=object))

def main():
	

	data = pd.read_csv("../hla.drb1:0102.mixedVal.swing.encoded.training210.csv",sep=",",header=0)
	max_len = 4500
	batch_size = 63
	indexed = sequences_to_indixes(data,max_len)

	indexed = torch.tensor(indexed)

	labels = list(data["Hit"])

	labels = torch.tensor(labels)

	
	eval(indexed,labels,batch_size,"HLA-DRB1:0102")

	#print("The AUROC for DRB1:0102 for mixed model: ",auc)

	data = pd.read_csv("../hla.drb1:0404.mixedVal.swing.encoded.training210.csv",sep=",",header=0)
        
	indexed = sequences_to_indixes(data,max_len)

	indexed = torch.tensor(indexed)

	labels = list(data["Hit"])

	labels = torch.tensor(labels)

	eval(indexed,labels,batch_size,"HLA-DRB1:0404")

	#print("The AUROC for DRB1:0404 for mixed model: ",auc)
	data = pd.read_csv("../hla.A02:02.mixedVal.swing.encoded.training210.csv",sep=",",header=0)
        
	indexed = sequences_to_indixes(data,max_len)
	indexed = torch.tensor(indexed)
	labels = list(data["Hit"])
	labels = torch.tensor(labels)

	eval(indexed,labels,batch_size,"HLA-A02:02")
	#print("The AUROC for A02:02 for mixed model: ",auc)

	data = pd.read_csv("../hla.B40:02.mixedVal.swing.encoded.training210.csv",sep=",",header=0)

	indexed = sequences_to_indixes(data,max_len)

	indexed = torch.tensor(indexed)

	labels = list(data["Hit"])

	labels = torch.tensor(labels)

	eval(indexed,labels,batch_size,"HLA-B40:02")

	#print("The AUROC for B40:02 for mixed model: ",auc)

	data = pd.read_csv("../hla.C05:01.mixedVal.swing.encoded.training210.csv",sep=",",header=0)

	indexed = sequences_to_indixes(data,max_len)

	indexed = torch.tensor(indexed)
	
	labels = list(data["Hit"])
	
	labels = torch.tensor(labels)

	eval(indexed,labels,batch_size,"HLA-C05:01")

	#print("The AUROC for C05:01 for mixed model: ",auc)

	data = pd.read_csv("../iag7.swing.encoded.training210.csv",sep=",",header=0)

	indexed = sequences_to_indixes(data,max_len)

	indexed = torch.tensor(indexed)

	labels = list(data["Hit"])

	labels = torch.tensor(labels)

	eval(indexed,labels,batch_size,"IAg7")

	#print("The AUROC for IAG7 for mixed model: ",auc)

	
	data = pd.read_csv("../iek.swing.encoded.training210.csv",sep=",",header=0)

	indexed = sequences_to_indixes(data,max_len)

	indexed = torch.tensor(indexed)

	labels = list(data["Hit"])

	labels = torch.tensor(labels)

	eval(indexed,labels,batch_size,"IEk")
	#print("The AUROC for IEk for mixed model: ",auc)

	print("Done!")


main()

