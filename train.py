import torch 
import torchtext
import numpy as np
from torchtext import datasets,data
import torch.nn.functional as F
def train(model,train_iter,val_iter,args): 
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(),lr=args["lr"])
  best_acc = 0
  epochs = args["epochs"]
  log_interval = args["log_interval"]
  val_interval = args["val_interval"]
  if(args["cuda"]):
    model = model.to("cuda")
  for epoch in range(epochs):
    for step, batch in enumerate(train_iter):
      x = batch.text
      x = x[0]
      y = batch.label
      if(args["cuda"]):
        x = x.to("cuda")
        y = y.to("cuda")
      optimizer.zero_grad()
      y_pred = model(x)
      loss = criterion(y_pred,y)
      loss.backward()
      optimizer.step()

      if(step%log_interval==0):
        acc = int((y_pred.argmax(dim=1)==y).sum())/y.shape[0]
        if(args["log"]==True):
          print("EPOCH{},STEP{}--Loss:{:.6f},ACC:{:.6f}".
                format(epoch,step,loss.item(),acc))
      if(step%val_interval==0):
        val_loss,val_acc = eval(model,val_iter,args)
        if(val_acc>best_acc):
          if(args["rand"]):
            name = "rand"
          elif(args["static"]):
            name = "static"
          else:
            name = "non_static"
          torch.save(model.state_dict(), name+"_model.pt")

def eval(model,val_iter,args):
  avg_loss = 0
  acc = 0
  batch_num = 0
  for batch in  val_iter:
    x = batch.text 
    x = x[0]
    y = batch.label
    if(args["cuda"]):
      x = x.to(device = 'cuda')
      y = y.to(device = 'cuda')
    y_pred = model(x)
    avg_loss += F.cross_entropy(y_pred,y).item()
    acc += int((y_pred.argmax(dim=1)==y).sum())
    batch_num += 1

  avg_loss/= batch_num
  acc/=len(val_iter.dataset)
  if(args["log"]==True):
    print("Loss:{:.6f}, ACC:{:.6f} over validation set".
              format(avg_loss,acc))
  return avg_loss,acc
