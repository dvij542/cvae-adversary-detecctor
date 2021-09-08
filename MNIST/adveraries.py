import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

def get_recon(models,X,Y) :
    recons = []
    for k in range(X.shape[0]) :
        recon,_,_,_ = models[Y[k]](X[k:(k+1)])
        recons.append(recon)
    recons = torch.cat(recons)
    return recons


def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return X + epsilon * delta.grad.detach().sign()

def fgsm_L2(model, X, y, epsilon):
    """ Construct FGSM-L2 adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return X + epsilon * delta.grad.detach()/torch.norm(torch.norm((delta.grad.detach()),dim=2,keepdim=True),dim=3,keepdim=True)

def R_fgsm(model, X, y, epsilon, alpha):
    """ Construct R-FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    return fgsm(model, random(X,alpha),y,epsilon)

def R_fgsm_L2(model, X, y, epsilon, alpha):
    """ Construct R-FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    return fgsm_L2(model, random(X,alpha), y, epsilon)

def BIM(model, X, y, epsilon, epsilon_step, no_of_steps):
    """ Construct BIM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    Xi = X.clone()
    for i in range(no_of_steps) :
      # delta = torch.zeros_like(X, requires_grad=True)
      loss = nn.CrossEntropyLoss()(model(X + delta), y)
      loss.backward()
      X = torch.clip(X.clone() + epsilon_step * delta.grad.detach().sign(),-1.,1.)
    diff = X - Xi
    return Xi + torch.clip(diff,-epsilon,epsilon)

def BIM_L2(model, X, y, epsilon, epsilon_step, no_of_steps):
    """ Construct BIM-L2 adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    Xi = X.clone()
    for i in range(no_of_steps) :
          loss = nn.CrossEntropyLoss()(model(X + delta), y)
      loss.backward()
      X = torch.clip(X.clone() + epsilon_step * (delta.grad.detach())/torch.norm((delta.grad.detach()),dim=1,keepdim=True),-1.,1.)
    diff = X - Xi
    factor = torch.clip(torch.norm(torch.norm((diff.detach()),dim=2,keepdim=True),\
              dim=3,keepdim=True),0,epsilon)\
              / torch.norm(torch.norm((diff.detach()),dim=2,keepdim=True),\
              dim=3,keepdim=True)
    return Xi + diff*factor

def random(X, epsilon) :
    delta = torch.rand_like(X).to(DEVICE) - .5
    return torch.clip(X + epsilon*delta,0.,1.)

def CW(model,X,y,epsilon,epsilon_step,no_of_steps,c,target):
    """ Construct CW adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    Xn = X.clone()
    # Iterations
    for i in range(no_of_steps) :
      sec_target = []
      A = model(Xn+delta)
      for i in range(A.shape[0]) :
        maxi = -1
        maxval = -10000
        for j in range(A.shape[1]) :
          if A[i,j] > maxval and j!=target[i]:
            maxval = A[i,j]
            maxi = j
        sec_target.append(maxi)
      
      val = torch.diag(A[:,target[:]])
      val_targ = torch.diag(A[:,sec_target[:]])
      # print(val)
      # print(val_targ)
      # for i in range(A.shape[0]) :  
      #   A[i,target[i]] = -1000
      loss = -torch.mean((Xn+delta-X)**2) - torch.mean(c*torch.clip(val_targ-val,-4,1000))
      loss.backward()
      Xn = Xn.clone() + epsilon_step * delta.grad.detach()
    diff = Xn - X
    return X + torch.clip(diff,-epsilon,epsilon) 

def S_BIM(model,model_detector,X,target,epsilon,sigma,epsilon_step,no_of_steps):
    """ Construct S-BIM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    delta_detector = torch.zeros_like(X, requires_grad=True)
    Xn = X.clone()
    for i in range(no_of_steps) :
      sec_target = []
      A = model(Xn+delta)
      for i in range(A.shape[0]) :
        maxi = -1
        maxval = -10000
        for j in range(A.shape[1]) :
          if A[i,j] > maxval and j!=target[i]:
            maxval = A[i,j]
            maxi = j
        sec_target.append(maxi)
      
      val = torch.diag(A[:,target[:]])
      val_targ = torch.diag(A[:,sec_target[:]])
      loss = -sigma*torch.mean((Xn+delta-get_recon(model_detector,Xn+delta,target))**2) - (1-sigma)*torch.mean(torch.clip(val_targ-val,-4,1000))
      loss.backward()
      Xn = Xn.clone() + epsilon_step * delta.grad.detach().sign()
    diff = Xn - X
    return X + torch.clip(diff,-epsilon,epsilon) 



from torch.autograd import Variable
from utee import selector
model_raw, ds_fetcher, is_imagenet = selector.select('cifar10')