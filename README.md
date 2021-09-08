# cvae-adversary-detecctor
CVAE based Adversary detection algorithm for image classification networks

# Install dependencies
`git clone https://github.com/aaron-xichen/pytorch-playground.git`

`cd pytorch-playground`

`python setup.py develop --user`

`pip install -U scikit-learn`

`pip install pandas`

# Train
Navigate to the dataset folder (MNIST/CIFAR-10). Run the following script, trained models will me saved to 'models/' directory

`python train.py`

# Test 
The trained models present in 'models/' directory of the dataset is used for adversary detection. The ROC curve is used as a statistic for differentiating adversaries. Run the below script

`python test.py`


