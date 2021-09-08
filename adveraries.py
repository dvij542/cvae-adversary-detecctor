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
    return torch.clip(X + epsilon * delta.grad.detach().sign(),-1.,1.)

def fgsm_L2(model, X, y, epsilon):
    """ Construct FGSM-L2 adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return torch.clip(X + epsilon * delta.grad.detach()/torch.norm((delta.grad.detach()),dim=1,keepdim=True),-1.,1.)

def R_fgsm(model, X, y, epsilon, alpha):
    """ Construct R-FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    return torch.clip(fgsm(model, random(X,alpha), y, epsilon),-1.,1.)

def R_fgsm_L2(model, X, y, epsilon, alpha):
    """ Construct R-FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    return torch.clip(fgsm_L2(model, random(X,alpha), y, epsilon),-1.,1.)

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

def random(X, epsilon) :
    delta = 2*torch.rand_like(X).to(DEVICE) - 1
    return torch.clip(X + epsilon*delta,0.,1.)

def BIM_L2(model, X, y, epsilon, epsilon_step, no_of_steps): # Also called as BIM
    """ Construct BIM-L2 adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    Xi = X.clone()
    for i in range(no_of_steps) :
      loss = nn.CrossEntropyLoss()(model(X + delta), y)
      loss.backward()
      X = torch.clip(X.clone() + epsilon_step * (delta.grad.detach())/torch.norm((delta.grad.detach()),dim=1,keepdim=True),-1.,1.)
    diff = X - Xi
    return Xi + torch.clip(diff,-epsilon,epsilon)

def CW(model,X,y,epsilon,epsilon_step,no_of_steps,c,target):
    """ Construct CW adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    Xn = X.clone()

    # Iterations
    for i in range(no_of_steps) :
      A = model(Xn+delta)
      val = torch.tensor([A[i,target[i]] for i in range(A.shape[0])]).to(DEVICE)
      for i in range(A.shape[0]) :  
        A[i,target[i]] = -1000
      loss = -torch.mean((Xn+delta-X)**2) - torch.mean(c*torch.clip(torch.max(A,dim=1).values-val,-4,1000))
      loss.backward()
      Xn = Xn.clone() + epsilon_step * delta.grad.detach() / torch.norm((delta.grad.detach()),dim=1,keepdim=True)
    diff = Xn - X
    return X + torch.clip(diff,-epsilon,epsilon) 

def S_BIM(model,model_detector,X,target,epsilon,sigma,epsilon_step,no_of_steps):
    """ Construct S-BIM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    delta_detector = torch.zeros_like(X, requires_grad=True)
    Xn = X.clone()
    for i in range(no_of_steps) :
      A = model(X+delta)
      val = torch.tensor([A[j,target[j]] for j in range(A.shape[0])]).to(DEVICE)
      for j in range(A.shape[0]) :  
        A[j,target[j]] = -1000
      loss = -sigma*torch.mean((Xn+delta-get_recon(model_detector,Xn+delta,target))**2) - (1-sigma)*torch.mean(torch.clip(torch.max(A,dim=1).values-val,-4,1000))
      loss.backward()
      Xn = Xn.clone() + epsilon_step * delta.grad.detach() / torch.norm((delta.grad.detach()),dim=1,keepdim=True)
    diff = Xn - X
    return X + torch.clip(diff,-epsilon,epsilon) 



from torch.autograd import Variable
from utee import selector
model_raw, ds_fetcher, is_imagenet = selector.select('cifar10')