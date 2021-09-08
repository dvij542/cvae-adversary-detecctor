from torch.autograd import Variable
from utee import selector
from torchvision.datasets import MNIST,CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch 
from model import *
import numpy as np
from adveraries import *
model_raw, ds_fetcher, is_imagenet = selector.select('cifar10')

mnist_transform = transforms.Compose([
        transforms.ToTensor(),
])

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")
print(DEVICE)
dataset_path = '~/datasets'
batch_size = 128


def get_same_index(target, label):
    label_indices = []
    for i in range(len(target)):
        if target[i] == label:
            label_indices.append(i)
    return label_indices

train_dataset_raw = CIFAR10(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset_raw  = CIFAR10(dataset_path, transform=mnist_transform, train=False, download=True)
print(len(test_dataset_raw))
# train_dataset_raw = train_dataset_raw[:30000]
# print(len(train_dataset_raw))
kwargs = {} 
train_loader = DataLoader(dataset=train_dataset_raw, batch_size=batch_size, shuffle=True, **kwargs)
test_loader1  = DataLoader(dataset=test_dataset_raw,  batch_size=batch_size, shuffle=True,  **kwargs)
label_class = 1# birds

# Get indices of label_class
# train_indices = get_same_index(train_dataset_raw, label_class)

# bird_set = torch.utils.data.Subset(train_dataset_raw, train_indices)

models = []
avg_recon_error = []
for clas in range(10) :
    print(clas)
    model = VAE_conv(
            label='Cifar10',
            image_size=32,
            channel_num=3,
            kernel_num=128,
            z_size=128,
            conditional=False,
            n_labels=10
        ).to(DEVICE)
    model.load_state_dict(torch.load('models_trained/curr_t'+str(clas)+'.pt', map_location=torch.device('cpu')))
    models.append(model)
    train_indices = get_same_index(train_dataset_raw.targets, clas)
    bird_set = torch.utils.data.Subset(train_dataset_raw, train_indices)
    train_loader_dash = torch.utils.data.DataLoader(dataset=bird_set, shuffle=True,
                                           batch_size=128, drop_last=True)
    for iteration, (x, y) in enumerate(train_loader_dash):
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_dash,_,_,_ = model(x)
        avg_recon_error.append(torch.nn.functional.binary_cross_entropy(
        x_dash.view(-1, 32*32*3), x.view(-1, 32*32*3),reduction='mean').detach().cpu().numpy())
        break
avg_recon_error = torch.tensor(np.array(avg_recon_error)).unsqueeze(1).to(DEVICE)

# Experiments

np.random.seed(1)
import matplotlib.pyplot as plt

types = ['normal','random','fgsm','fgsm-L2','R-fgsm','R-fgsm-L2','BIM','BIM-L2','CW','S-BIM']          
data_loader = DataLoader(
    dataset=test_dataset_raw, batch_size=16, shuffle=True)

recons_errors = {}
no_of_incorrects = {}
no_of_corrects = {}
for t in types :
  recons_errors[t] = []
  no_of_incorrects[t] = 0
  no_of_corrects[t] = 0
model.eval()
epsilon = 20/255
epsilon_step = 4/255
sigma = 0.5
no_of_iters = 10
epsilon_random = 2/255

for iteration, (x, y) in enumerate(tqdm(data_loader,position=0,leave=True)):
    x, y = x.to(DEVICE), y.to(DEVICE)
    for t in types :
        if t == 'normal' :
          x_type = x
        if t == 'random' :
          x_type = random(x,epsilon/2)
        if t == 'fgsm' :
          x_type = fgsm(model_raw,x,y,epsilon)
        if t == 'fgsm-L2' :
          x_type = fgsm_L2(model_raw,x,y,2*epsilon)
        if t == 'R-fgsm' :
          x_type = R_fgsm(model_raw,x,y,epsilon,epsilon_random)
        if t == 'R-fgsm-L2' :
          x_type = R_fgsm_L2(model_raw,x,y,2*epsilon,epsilon_random)
        if t == 'BIM' :
          x_type = BIM(model_raw,x,y,epsilon,epsilon_step,no_of_iters)
        if t == 'BIM-L2' :
          x_type = BIM_L2(model_raw,x,y,epsilon,epsilon_step,no_of_iters)
        if t == 'CW' :
          rands = torch.randint(0,9,(x.shape[0],)).to(DEVICE)
          x_type = CW(model_raw,x,y,epsilon,epsilon_step,no_of_iters,2,(rands<y)*rands + (rands>=y)*(rands+1)) 
        if t == 'S-BIM' :
          rands = torch.randint(0,9,(x.shape[0],)).to(DEVICE)
          x_type = S_BIM(model_raw,models,x,(rands<y)*rands + (rands>=y)*(rands+1),epsilon,sigma,epsilon_step,no_of_iters//2)

        y_type = torch.argmax(model_raw(x_type),1)
        no_of_incorrects[t] += torch.sum((y_type!=y))
        no_of_corrects[t] += torch.sum((y_type==y))
        if t!='normal' and t!='random' :
            indexes = (y_type!=y).nonzero()
            x_type = x_type[indexes[:,0],:,:,:]
            y_type = y_type[indexes[:,0]]
        else: 
            indexes = (y_type==y).nonzero()
            x_type = x_type[indexes[:,0],:,:,:]
            y_type = y_type[indexes[:,0]]
        
        recon_x_type = get_recon(models,x_type, y_type)
        if x_type.shape[0]!=0:
            recon_x_type = get_recon(models,x_type, y_type)
            recons_errors[t].append(torch.mean((recon_x_type.detach().view(-1, 32*32*3)-x_type.detach().view(-1, 32*32*3))**2,dim=1))

for t in types :
    recons_errors[t] = torch.cat(recons_errors[t],dim=0).cpu().numpy()

types =         ['normal','random','fgsm' ,'fgsm-L2','R-fgsm','R-fgsm-L2','BIM'   ,'BIM-L2','CW'  ,'S-BIM']          
type_colors_l = ['blue'  ,'red'   ,'brown','yellow' ,'green' ,'purple'   ,'orange','violet','grey','pink' ]
itr = 0
type_color = {}
for t in types :
  type_color[t] = type_colors_l[itr]
  itr += 1 

# np.savetxt('recons_errors.txt',recons_errors)
import pickle
geeky_file = open('recons_errors.txt', 'wb')
pickle.dump(recons_errors, geeky_file)
geeky_file.close()

# types = ['normal','S-BIM'] 
import matplotlib.pyplot as plt
ref = []

for iteration, (x, y) in enumerate(tqdm(data_loader,position=0,leave=True)):
    x_type, y = x.to(DEVICE), y.to(DEVICE)
    y_type = torch.argmax(model_raw(x_type),1)
    recon_x_type = get_recon(models,x_type, y_type)
    ref.append((torch.mean((recon_x_type.detach().view(-1, 32*32*3)-x_type.detach().view(-1, 32*32*3))**2,dim=1)))

ref = torch.cat(ref,dim=0).cpu().numpy()
n, bins, patches = plt.hist(x=ref, bins = np.arange(0.0,0.08,0.001) , color='brown',
                              alpha=0.5,label='train images')
types =         ['normal','random','fgsm' ,'fgsm-L2','R-fgsm','R-fgsm-L2','BIM'   ,'BIM-L2','CW'  ,'S-BIM']          

for t in types :
  n, bins, patches = plt.hist(x=recons_errors[t], bins = np.arange(0.005,0.035,0.0005) , color=type_color[t],
                              alpha=0.5,label=t)
  print("Ratio for " + t + " : ", no_of_incorrects[t]/(no_of_incorrects[t] + no_of_corrects[t]))
  print("No of corrects for " + t + " : ", no_of_corrects[t])

plt.grid(axis='y')
# size = plt.get_size_inches()*plt.dpi
# print(size)
plt.xlabel('Reconstruction error')
plt.ylabel('# of images')
plt.legend()
plt.title('Adversaries Vs normal examples reconstruction error')
plt.savefig('recons-errors.png')
maxfreq = n.max()
plt.show()

pvals = {}
# ref = np.array(recons_errors['normal'])
# types = ['normal','random','fgsm','fgsm','fgsmR','fgsmR','BIM','fgsmI','CW','CW']          
for t in types :
  pvals[t] = []
  for error in recons_errors[t] :
    val = np.sum(ref>error)/len(ref)#.shape[0]
    pvals[t].append(val)

plt.figure(figsize=(30, 16))
plt.tight_layout()
p = 0
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
for t in types :
  p += 1
  plt.subplot(2, 5, p)
  n, bins, patches = plt.hist(x=pvals[t], bins=np.arange(0.,1.,0.05), color=type_color[t],
                              alpha=0.7)
  plt.title(t,fontsize = 40)
  plt.grid(axis='y')
  plt.xlabel('p-values',fontsize = 20)
  plt.ylabel('# of images',fontsize = 20)

plt.savefig('p-values.png')
plt.show()  

from sklearn import metrics
tpr = {}
for t in types :
  tpr[t] = []
  pvals[t] = np.array(pvals[t])
  # auc = 0
  fpr = []
  for pval in np.arange(0,1.01,0.01) :
    tpr[t].append(np.sum(pvals[t]<pval)/pvals[t].shape[0])
    # auc += np.sum(pvals[t]<pval)/pvals[t].shape[0]*0.01

  
  auc = metrics.auc(tpr['normal'], tpr[t])#auc - 0.5*(np.sum(pvals[t]<0)/pvals[t].shape[0]-np.sum(pvals[t]<1)/pvals[t].shape[0])*0.01
  plt.plot(tpr['normal'],tpr[t],color=type_color[t],label=t)
  print("AUC for " + t + ": ", str(auc))

plt.legend()
plt.savefig('linear_comparison.png')
plt.show()

tpr = {}
for t in types :
  tpr[t] = []
  pvals[t] = np.array(pvals[t])
  for fpr in np.arange(0.01,1,0.01) :
    tpr[t].append(np.sum(pvals[t]<fpr)/pvals[t].shape[0])
  plt.plot(np.arange(0.01,1,0.01),tpr[t],color=type_color[t],label=t)

# types = ['normal','random','fgsm','fgsm-L2','R-fgsm','R-fgsm-L2','BIM','BIM-L2','CW','target_CW2']          

plt.legend()
plt.xscale('log')
plt.savefig('log_comparison.png')
plt.show()



