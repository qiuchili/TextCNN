import torch
import torch.nn as nn
import torch.nn.functional as F
class textCNN(nn.Module):
  def __init__(self,args):
    
    super(textCNN,self).__init__()
    vec_dim = args["vec_dim"]
    embedding_matrix = args["embedding_matrix"]
    Ci = 1
    Co = args["class_num"]
    features = args["features_per_size"]
    kernel_sizes = args["kernel_sizes"]
    dropout_rate = args["dropout_rate"]
    if(args["rand"]):
      self.embed = nn.Embedding(args['vocab_size'],args["vec_dim"])
    else:
      if(args["static"]):
        self.embed = nn.Embedding.from_pretrained(embedding_matrix)
      else:
        self.embed = nn.Embedding.from_pretrained(embedding_matrix,freeze=False)
    
    self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,out_channels=feature,kernel_size=(size,vec_dim),padding = (size-1,0))for (size,feature ) in zip(kernel_sizes,features) ])
    self.dropout = nn.Dropout(args["dropout_rate"])
    
    self.out =  nn.Linear(sum(features),Co)
    
  def forward(self,x):
    x = self.embed(x)
    x = x.unsqueeze(1)
    x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
    x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
    x = torch.cat(x,1)
    x = self.dropout(x)
    x = self.out(x)
    return x
