## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
#         self.conv1 = nn.Conv2d(1, 32, 5)
                
        #JEN_NOTE: 96x96 grayscale image according to ref, but the transformation implementation is 224x224 --> let's go with 96x96
        ## output size = (W-F)/S +1 = (96-5)/1 +1 = 92
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        
            
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=256*12*12, out_features=1000) 
        self.fc2 = nn.Linear(in_features=1000, out_features=1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=136)
        
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Convolution Layers = Convolution --> Activation --> Pooling --> Dropout

        x = self.drop1(self.pool(F.relu(self.conv1(x))))
#         print("Layer 1...", x.shape)

        x = self.drop2(self.pool(F.relu(self.conv2(x))))
#         print("Layer 2...", x.shape)

        x = self.drop3(self.pool(F.relu(self.conv3(x))))
#         print("Layer 3...", x.shape)

        x = self.drop4(self.pool(F.relu(self.conv4(x))))
#         print("Layer 4...", x.shape)

        # Flatten layer
        x = x.view(x.size(0), -1)
        #print("Flatten size: ", x.shape)

        
        # Dense Layers = Dense --> Activation --> Dropout
        
        x = self.drop5(F.relu(self.fc1(x)))
#         print("First dense size: ", x.shape)

        x = self.drop6(F.relu(self.fc2(x)))
#         print("Second dense size: ", x.shape)

        x = self.fc3(x)
#         print("Final dense size: ", x.shape)


        # a modified x, having gone through all the layers of your model, should be returned
        return x
