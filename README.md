# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

PART 2:
1. Train with vgg16
> python train.py flowers --gpu True --save_dir checkpoint_vgg16.pth
Data directory     = flowers
Save directory     = checkpoint_vgg16.pth
Architecture       = vgg16
Learning Rate      = 0.001
Hidden Unit        = [25088, 4096]
Epochs             = 3
GPU Available      = True

Train the classifier:
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.torch/models/vgg16-397923af.pth
100%|██████████████████████████████████████████████████████████████████████████████████████| 553433881/553433881 [00:28<00:00, 19689320.17it/s]
Epoch: 1/3 ...  Train loss: 5.157 ...  Valid loss: 1.938 ...  Valid accuracy: 0.528 ... 
Epoch: 1/3 ...  Train loss: 1.973 ...  Valid loss: 1.043 ...  Valid accuracy: 0.710 ... 
Epoch: 1/3 ...  Train loss: 1.464 ...  Valid loss: 0.832 ...  Valid accuracy: 0.766 ... 
Epoch: 2/3 ...  Train loss: 1.287 ...  Valid loss: 0.642 ...  Valid accuracy: 0.810 ... 
Epoch: 2/3 ...  Train loss: 1.049 ...  Valid loss: 0.625 ...  Valid accuracy: 0.824 ... 
Epoch: 2/3 ...  Train loss: 1.017 ...  Valid loss: 0.526 ...  Valid accuracy: 0.865 ... 
Epoch: 3/3 ...  Train loss: 1.037 ...  Valid loss: 0.511 ...  Valid accuracy: 0.856 ... 
Epoch: 3/3 ...  Train loss: 0.874 ...  Valid loss: 0.611 ...  Valid accuracy: 0.834 ... 
Epoch: 3/3 ...  Train loss: 0.895 ...  Valid loss: 0.489 ...  Valid accuracy: 0.865 ... 
Epoch: 3/3 ...  Train loss: 0.862 ...  Valid loss: 0.456 ...  Valid accuracy: 0.875 ... 

Test the classifier:
Test loss: 0.608 ...  Test accuracy: 0.827 ... 

Check point saved in: checkpoint_vgg16.pth

1. Train with densenet121
> python train.py flowers --arch densenet121 --hidden_units 1024 512 --save_dir checkpoint_densenet121.pth --gpu True
Data directory     = flowers
Save directory     = checkpoint_densenet121.pth
Architecture       = densenet121
Learning Rate      = 0.001
Hidden Unit        = [1024, 512]
Epochs             = 3
GPU Available      = True

Train the classifier:
/opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
Downloading: "https://download.pytorch.org/models/densenet121-a639ec97.pth" to /root/.torch/models/densenet121-a639ec97.pth
100%|████████████████████████████████████████████████████████████████████████████████████████| 32342954/32342954 [00:00<00:00, 37455552.06it/s]
Epoch: 1/3 ...  Train loss: 4.268 ...  Valid loss: 3.643 ...  Valid accuracy: 0.278 ... 
Epoch: 1/3 ...  Train loss: 3.315 ...  Valid loss: 2.510 ...  Valid accuracy: 0.497 ... 
Epoch: 1/3 ...  Train loss: 2.388 ...  Valid loss: 1.636 ...  Valid accuracy: 0.677 ... 
Epoch: 2/3 ...  Train loss: 1.774 ...  Valid loss: 1.250 ...  Valid accuracy: 0.708 ... 
Epoch: 2/3 ...  Train loss: 1.420 ...  Valid loss: 0.927 ...  Valid accuracy: 0.809 ... 
Epoch: 2/3 ...  Train loss: 1.207 ...  Valid loss: 0.780 ...  Valid accuracy: 0.820 ... 
Epoch: 3/3 ...  Train loss: 1.046 ...  Valid loss: 0.677 ...  Valid accuracy: 0.852 ... 
Epoch: 3/3 ...  Train loss: 0.959 ...  Valid loss: 0.585 ...  Valid accuracy: 0.869 ... 
Epoch: 3/3 ...  Train loss: 0.892 ...  Valid loss: 0.548 ...  Valid accuracy: 0.869 ... 
Epoch: 3/3 ...  Train loss: 0.854 ...  Valid loss: 0.503 ...  Valid accuracy: 0.882 ... 

Test the classifier:
Test loss: 0.499 ...  Test accuracy: 0.886 ... 

Check point saved in: checkpoint_densenet121.pth

2. Predict with vgg16
> python predict.py flower_to_predict.jpg checkpoint_vgg16.pth --gpu True
Image directory       = flower_to_predict.jpg
Checkpoint directory  = checkpoint_vgg16.pth
Top K                 = 5
Category Names        = cat_to_name.json
GPU Available         = True

Prediction with the following model:
vgg16
Sequential(
  (0): Linear(in_features=25088, out_features=4096, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.2)
  (3): Linear(in_features=4096, out_features=102, bias=True)
  (4): LogSoftmax()
)

Result:

Predicted flower name: sunflower with a probability of: 98.17%

The top 5 most likely classes are:

sunflower with 98.17%
blanket flower with 1.80%
gazania with 0.02%
black-eyed susan with 0.01%
english marigold with 0.00%

2. Predict with densenet121
> python predict.py flower_to_predict.jpg checkpoint_densenet121.pth --gpu True
Image directory       = flower_to_predict.jpg
Checkpoint directory  = checkpoint_densenet121.pth
Top K                 = 5
Category Names        = cat_to_name.json
GPU Available         = True

Prediction with the following model:
/opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
densenet121
Sequential(
  (0): Linear(in_features=1024, out_features=512, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.2)
  (3): Linear(in_features=512, out_features=102, bias=True)
  (4): LogSoftmax()
)

Result:

Predicted flower name: sunflower with a probability of: 77.56%

The top 5 most likely classes are:

sunflower with 77.56%
colt's foot with 7.59%
blanket flower with 3.95%
black-eyed susan with 3.19%
common dandelion with 2.26%
