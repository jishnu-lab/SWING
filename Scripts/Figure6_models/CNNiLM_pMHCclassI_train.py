import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
from sklearn.metrics import roc_auc_score 
from sklearn.model_selection import StratifiedKFold
import pandas as pd 
import warnings
import argparse 
import wandb 

# Suppress all warnings from PyTorch
warnings.filterwarnings("ignore")


class ilmCNN(nn.Module):
	def __init__(self,input_dim, embedding_dim,num_classes,num_kernels,kernel_size,dropout):
		super(ilmCNN,self).__init__()
		self.embedding = nn.Embedding(input_dim,embedding_dim)	# embedding for each char in the ILM
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

def sequences_to_indixes(seq,max_length):
	indices = []
	for item in seq:
		indices.append(int(item))

	if len(indices) < max_length:
		indices += [10] * (max_length - len(indices)) # padding 

	else:
		indices = indices[:max_length]

	return indices 

# stratified cross validation with AUROC

def train(model,X,y,num_classes,input_dim,embedding_dim,lr,num_kernels,epochs,kernel_size,dropout,batch_size):
	
	train_dataset = torch.utils.data.TensorDataset(torch.tensor(X),torch.tensor(y))
	train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

	model = ilmCNN(input_dim,embedding_dim,num_classes,num_kernels,kernel_size,dropout)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(),lr=lr)
	# train the model
	for epoch in range(epochs):
		model.train()
		epoch_loss = 0.0
		for inputs, labels in train_loader:
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs,labels)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
		epoch_loss /= len(train_loader)
		print(f'Epoch [{epoch + 1}/{epochs}] Average Loss: {epoch_loss:.4f}')

	return model 
		
def main():


	#commandline args for the input and the output files 
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--lr",type=float,help="The learning rate")
	parser.add_argument("--num_kernels",type=int,help="The number of kernels")
	parser.add_argument("--epochs",type=int,help="The number of epochs for the model")
	parser.add_argument("--kernel_size",type=int,help="The size of the kernel")
	parser.add_argument("--dropout",type=float,help="The dropout rate")
	parser.add_argument("--embedding_dim",type=int,help="The dimensions of the embeddings")
	parser.add_argument("--batch_size",type=int,help="The batch size for the model")
	#parser.add_argument("--run_name",type=str,help="The run name")
	#parser.add_argument("--project_name",type=str,help="The name of the project")

	args = parser.parse_args()

	#wandb.init(project=args.project_name, name = args.run_name, config = args)

	
	input_dim = 11
	#embedding_dim = 128
	num_classes = 2
	max_len = 4500

	data = pd.read_csv("../classI.swing.encoded.training210.csv",sep=",",header=0)
	encodings = list(data["Encoded"])

	indexed = [sequences_to_indixes(seq, max_len) for seq in encodings]

	indexed = torch.tensor(indexed)

	labels = list(data["Hit"])

	labels = torch.tensor(labels)

	model  = train(ilmCNN, indexed, labels,num_classes,input_dim,args.embedding_dim,args.lr,args.num_kernels,args.epochs,args.kernel_size,args.dropout,args.batch_size)

	torch.save(model,"classI.pmhc.210.trained.1dCNN.pth") # saving the model 

	print("Done!")

	#wandb.log({"AUROC":auroc,"lr":args.lr,"num_kernels":args.num_kernels,"epochs":args.epochs,"kernel_size":args.kernel_size,"dropout":args.dropout,"embedding_dim":args.embedding_dim,"batch_size":args.batch_size})
	

main()


