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
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")
print(DEVICE)
dataset_path = '~/datasets'
batch_size = 128
model_raw, ds_fetcher, is_imagenet = selector.select('mnist')

mnist_transform = transforms.Compose([
        transforms.ToTensor(),
])
kwargs = {} 

train_dataset_raw = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset_raw  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)
print(len(test_dataset_raw))
# train_dataset_raw = train_dataset_raw[:30000]
print(len(train_dataset_raw))
train_loader = DataLoader(dataset=train_dataset_raw, batch_size=batch_size, shuffle=True, **kwargs)
test_loader1  = DataLoader(dataset=test_dataset_raw,  batch_size=batch_size, shuffle=True,  **kwargs)
label_class = 1 # ones

# Get indices of label_class
# train_indices = get_same_index(train_dataset_raw, label_class)

# one_set = torch.utils.data.Subset(train_dataset_raw, train_indices)

def get_same_index(target, label):
    label_indices = []
    for i in range(len(target)):
        if target[i] == label:
            label_indices.append(i)
    return label_indices

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x.view(-1, 28*28*1), x.view(-1, 28*28*1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = VAE(
#     encoder_layer_sizes=encoder_layer_sizes,
#     latent_size=latent_size,
#     decoder_layer_sizes=decoder_layer_sizes,
#     conditional=conditional,
#     num_labels=n_labels if conditional else 0).to(device)


print("Loading CVAE models :-")
models = []
avg_recon_error = []
for clas in range(10) :
    print("Loading model for class ", str(clas))
    model = VAE_conv(
            label='MNIST',
            image_size=28,
            channel_num=1,
            kernel_num=128,
            z_size=128,
            conditional=False,
            n_labels=10
        ).to(device)
    model.load_state_dict(torch.load('models_trained/curr_t'+str(clas)+'.pt', map_location=torch.device('cpu')))
    models.append(model)
    train_indices = get_same_index(train_dataset_raw.targets, clas)
    bird_set = torch.utils.data.Subset(train_dataset_raw, train_indices)
    train_loader_dash = torch.utils.data.DataLoader(dataset=bird_set, shuffle=True,
                                           batch_size=128, drop_last=True)
    for iteration, (x, y) in enumerate(train_loader_dash):
        x, y = x.to(device), y.to(device)
        x_dash,_,_,_ = model(x)
        avg_recon_error.append(torch.nn.functional.binary_cross_entropy(
        x_dash.view(-1, 28*28*1), x.view(-1, 28*28*1),reduction='mean').detach().cpu().numpy())
        break
avg_recon_error = torch.tensor(np.array(avg_recon_error)).unsqueeze(1).to(device)
# print(avg_recon_error)

# Experiments

np.random.seed(1)
torch.random.manual_seed(1)
import matplotlib.pyplot as plt

types = ['normal','random','fgsm','fgsm-L2','R-fgsm','R-fgsm-L2','BIM','BIM-L2','CW','S-BIM']          
data_loader = DataLoader(
    dataset=test_dataset_raw, batch_size=16, shuffle=True)

recons_errors = {}
no_of_incorrects = {}
no_of_corrects = {}
curr_i = {}
for t in types :
  recons_errors[t] = np.zeros(16*len(data_loader))
  no_of_incorrects[t] = 0
  no_of_corrects[t] = 0
  curr_i[t] = 0

model.eval()
epsilon = 20/255
epsilon_step = 4/255

epsilon1 = 500/255
epsilon_step1 = 60/255
sigma = 0.5
no_of_iters = 12
no_of_iters1 = 12
epsilon_random = 4/255

print("Getting reconstructions errors :- ")
for iteration, (x, y) in enumerate(tqdm(data_loader,position=0,leave=True)):
    x, y = x.to(device), y.to(device)
    for t in types :
        if t == 'normal' :
          x_type = x
        if t == 'random' :
          x_type = random(x,epsilon)
        if t == 'fgsm' :
          x_type = fgsm(model_raw,x,y,epsilon)
        if t == 'fgsm-L2' :
          x_type = fgsm_L2(model_raw,x,y,epsilon1)
        if t == 'R-fgsm' :
          x_type = R_fgsm(model_raw,x,y,epsilon,epsilon_random)
        if t == 'R-fgsm-L2' :
          x_type = R_fgsm_L2(model_raw,x,y,epsilon1,epsilon_random)
        if t == 'BIM' :
          x_type = BIM(model_raw,x,y,epsilon,epsilon_step,no_of_iters)
        if t == 'BIM-L2' :
          x_type = BIM_L2(model_raw,x,y,epsilon1,epsilon_step1,no_of_iters1)
        if t == 'CW' :
          rands = torch.randint(0,9,(x.shape[0],)).to(device)
          x_type = CW(model_raw,x,y,epsilon,3*epsilon_step,no_of_iters,2,(rands<y)*rands + (rands>=y)*(rands+1)) 
        if t == 'S-BIM' :
          rands = torch.randint(0,9,(x.shape[0],)).to(device)
          x_type = S_BIM(model_raw,models,x,(rands<y)*rands + (rands>=y)*(rands+1),epsilon,sigma,epsilon_step,no_of_iters)

        y_type = torch.argmax(model_raw(x_type),1)
        no_of_incorrects[t] += torch.sum((y_type!=y))
        no_of_corrects[t] += torch.sum((y_type==y))
        if t!='normal' and t!='random' :
            indexes = (y_type!=y).nonzero()
            x_type = x_type[indexes[:,0],:,:,:]
            y_type = y_type[indexes[:,0]]
        if x_type.shape[0]!=0:
            recon_x_type = get_recon(models,x_type, y_type)
            recons_errors[t][curr_i[t]:(curr_i[t]+x_type.shape[0])] = torch.mean((recon_x_type.detach().view(-1, 28*28*1)-x_type.detach().view(-1, 28*28*1))**2,dim=1).cpu().numpy()
            curr_i[t] += x_type.shape[0]
    # break

for t in types :
    recons_errors[t] = recons_errors[t][:curr_i[t]]

types =         ['normal','random','fgsm' ,'fgsm-L2','R-fgsm','R-fgsm-L2','BIM'   ,'BIM-L2','CW'  ,'S-BIM']          
type_colors_l = ['blue'  ,'red'   ,'brown','yellow' ,'green' ,'purple'   ,'orange','violet','grey','pink' ]
itr = 0
type_color = {}
for t in types :
  type_color[t] = type_colors_l[itr]
  itr += 1 

# np.savetxt('recons_errors.txt',recons_errors)
geeky_file = open('recons_errors.txt', 'wb')
pickle.dump(recons_errors, geeky_file)
geeky_file.close()


geeky_file = open('recons_errors.txt', 'rb')
# pickle.load()
recons_errors = pickle.load(geeky_file)
types =         ['normal','random','fgsm' ,'fgsm-L2','R-fgsm','R-fgsm-L2','BIM'   ,'BIM-L2','CW'  ,'S-BIM']          
type_colors_l = ['blue'  ,'red'   ,'brown','yellow' ,'green' ,'purple'   ,'orange','violet','grey','pink' ]
itr = 0
type_color = {}
for t in types :
  type_color[t] = type_colors_l[itr]
  itr += 1 


ref = []
print("Getting reference reconstruction errors :- ")
for iteration, (x, y) in enumerate(tqdm(train_loader,position=0,leave=True)):
    x_type, y = x.to(device), y.to(device)
    y_type = torch.argmax(model_raw(x_type),1)
    recon_x_type = get_recon(models,x_type, y_type)
    ref.append((torch.mean((recon_x_type.detach().view(-1, 28*28*1)-x_type.detach().view(-1, 28*28*1))**2,dim=1)))

ref = torch.cat(ref,dim=0).cpu().numpy()
# n, bins, patches = plt.hist(x=ref, bins = np.arange(0.0,0.08,0.001) , color='brown',
#                               alpha=0.5,label='train images')
types =         ['normal','random','fgsm' ,'fgsm-L2','R-fgsm','R-fgsm-L2','BIM'   ,'BIM-L2','CW'  ,'S-BIM']          

for t in types :
  n, bins, patches = plt.hist(x=recons_errors[t], bins = np.arange(0.0,0.08,0.001) , color=type_color[t],
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
# plt.show()

# p = 0
# for t in types :
#   p += 1
#   n, bins, patches = plt.hist(x=pvals[t], bins=np.arange(0.,1.,0.05), color=type_color[t],
#                               alpha=0.7)
#   plt.grid(axis='y')
#   plt.xlabel('p-values')
#   plt.ylabel('# of images')
#   plt.title(t)
#   plt.savefig('p-values-mnist-' + t + '.png')
#   plt.show()

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

types = ['normal','random','fgsm','fgsm-L2','R-fgsm','R-fgsm-L2','BIM','BIM-L2','CW','target_CW2']          

plt.legend()
plt.xscale('log')
plt.savefig('log_comparison.png')
plt.show()

