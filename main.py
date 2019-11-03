import torch 
import torchtext
import numpy as np
import torch.nn as nn
from torchtext import datasets,data
import torch.nn.functional as F
from model import*
from train import*

TEXT = data.Field(lower = True,batch_first= True,include_lengths=True
                  ,sequential= True)
LABEL = data.Field(sequential= False,unk_token=None)
train_set,val_set,test_set = datasets.SST.splits(TEXT,LABEL,fine_grained =True)
TEXT.build_vocab(train_set, vectors="glove.840B.300d")
LABEL.build_vocab(train_set,val,test)
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train_set, val_set, test_set), batch_sizes=(50, 128, 256),shuffle=True)
	
args={}
args['vocab_size']=len(TEXT.vocab)
args["vec_dim"]=300
args['class_num']=len(LABEL.vocab)
args['embedding_matrix']=TEXT.vocab.vectors
args["lr"]=1e-3
args["log"]=False
args['epochs']=200
args["val_interval"]=100
args["features_per_size"] = [100,100,100]
args["kernel_sizes"] = [3,4,5]
args["dropout_rate"] = 0.5
args["log_interval"] = 100
args["cuda"] = torch.cuda.is_available()
args["static"] =  False
args["rand"] = True


rand_model = textCNN(args)
train(rand_model,train_iter,val_iter,args)  
rand_model.load_state_dict(torch.load("rand_model.pt"))
_,test_acc = eval(rand_model,val_iter,args)
print("accuracy over test set:{:.6f}".
              format(test_acc))
			  
args["rand"]=False
args["static"] = True			  
static_model = textCNN(args)
train(static_model,train_iter,val_iter,args)  
static_model.load_state_dict(torch.load("static_model.pt"))
_,test_acc = eval(static_model,val_iter,args)
print("Accuracy on test set:{:.6f}".
              format(test_acc))
			  
		
args["static"]=False
non_static_model = textCNN(args)
train(non_static_model,train_iter,val_iter,args)  
non_static_model.load_state_dict(torch.load("non_static_model.pt"))
_,test_acc = eval(non_static_model,val_iter,args)
print("Accuracy on test set:{:.6f}".
              format(test_acc))
			  
