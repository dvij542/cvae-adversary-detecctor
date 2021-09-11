# cvae-adversary-detector
CVAE based Adversary detection algorithm for image classification networks

# Install dependencies
Execute these commands to install the required dependencies
`git clone https://github.com/aaron-xichen/pytorch-playground.git`
`cd pytorch-playground`
`python setup.py develop --user`
`pip install -U scikit-learn`
`pip install pandas`

# Navigate to the dataset folder to test on 
`cd <MNIST/CIFAR-10>`

# Test 
The trained models present in 'trained_models/' directory of the dataset is used for adversary detection. The ROC curve is used as a statistic for differentiating adversaries. Run the below script. Run the following script to reproduce the results 
`python test.py`

# Train
To train the model from scratch, navigate to the dataset folder (MNIST/CIFAR-10). Run the following script, trained models will me saved to 'trained_models/' directory. Change hyper-parameters from the header of train.py script. Well organized code for mutable parameters for the model architecture and bash options will be released when made public
`python train.py`

# NOTE
The files 'MNIST.ipynb' and 'CIFAR-10.ipynb' contain the jupyter notebook with same code along with output and also contains the example figures which are added in the paper. Publicly available code will be well organized and formatted 
