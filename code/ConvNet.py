import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    # Custom constructor for ConvNet
    def __init__(self, mode):
        # Initialize the underlying PyTorch module framework
        super(ConvNet, self).__init__()

        # Reference for dimensions of input MNIST images 
        MNIST_IMAGE_LENGTH = 28
        MNIST_IMAGE_WIDTH = 28
        
        # Select the forward pass function based on mode for the ConvNet.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            in_features = MNIST_IMAGE_LENGTH * MNIST_IMAGE_WIDTH # 784
            # Hidden layer, 784 input features -> 100 neurons
            self.fc1 = nn.Linear(in_features=in_features, out_features=100)
            # One output score for each digit class
            self.fc_out = nn.Linear(in_features=100, out_features=10)
            self.forward = self.model_1

        elif mode == 2:
            # First convolution, 1 grayscale input channel -> 40 feature maps
            self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=40,
                kernel_size=5,
                stride=1,
                padding=0
            )
            # Second convolution, 40 input feature maps -> 40 output feature maps
            self.conv2 = nn.Conv2d(
                in_channels=40,
                out_channels=40,
                kernel_size=5,
                stride=1,
                padding=0
            )
            # Pooling layer
            self.pool = nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
            # After both convolution and pooling stages: 40 4x4 feature maps
            feature_channels = 40
            feature_height = 4
            feature_width = 4

            in_features = feature_channels * feature_height * feature_width # 640
            # Hidden layer, 640 input features -> 100 neurons
            self.fc1 = nn.Linear(in_features=in_features, out_features=100)
            # One output score for each digit class
            self.fc_out = nn.Linear(in_features=100, out_features=10)
            self.forward = self.model_2

        elif mode == 3:
            # First convolution, 1 grayscale input channel -> 40 feature maps
            self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=40,
                kernel_size=5,
                stride=1,
                padding=0
            )
            # Second convolution, 40 input feature maps -> 40 output feature maps
            self.conv2 = nn.Conv2d(
                in_channels=40,
                out_channels=40,
                kernel_size = 5,
                stride=1,
                padding=0
            )
            # Pooling layer
            self.pool = nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
            # Convolution and pooling produces 40 4x4 feature maps
            feature_channels = 40
            feature_height = 4
            feature_width = 4
            in_features = feature_channels * feature_height * feature_width # 640
            # Hidden layer, 640 input features -> 100 neurons
            self.fc1 = nn.Linear(in_features=in_features, out_features=100)
            # Output layer: 100 neurons -> 10 digit scores
            self.fc_out = nn.Linear(in_features=100, out_features=10)
            self.forward = self.model_3

        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        # One fully connected hidden layer
        # X is a tensor consisting of: [Batch_Size, Channels, Height, Width]
        # Flatten [Batch_size, 1, 28, 28] tensor into a flat [Batch_size, 784] tensor
        X = torch.flatten(X, start_dim=1)
        # Pass X through fully connected hidden layer 1, then apply sigmoid
        X = torch.sigmoid(self.fc1(X))
        # Pass X through the fully connected output layer to obtain 10 logits
        fcl = self.fc_out(X)
        return  fcl

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected hidden layer
        
        # Pass X through the first convolution layer
        # (batch, 1, 28, 28) -> (batch, 40, 24, 24)
        X = self.conv1(X)
        X = torch.sigmoid(X)

        # Apply first pooling layer
        # (batch, 40, 24, 24) -> (batch, 40, 12, 12)
        X = self.pool(X)

        # Pass X through the second convolution
        # (batch, 40, 12, 12) -> (batch, 40, 8, 8)
        X = self.conv2(X)
        X = torch.sigmoid(X)

        # Apply second pooling layer
        # (batch, 40, 8, 8) -> (batch, 40, 4, 4)
        X = self.pool(X)

        # Flatten [Batch_size, 40, 4, 4] tensor into a flat [Batch_size, 640] tensor
        X = torch.flatten(X, start_dim=1)

        # Pass X through the hidden fully connected layer, then apply sigmoid
        # 640 features -> 100 neurons
        X = self.fc1(X)
        X = torch.sigmoid(X)

        # Output fully connected layer
        # 100 neurons -> 10 digit scores
        fcl = self.fc_out(X)
        return fcl

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        
        # Pass X through the first convolution layer
        # (batch, 1, 28, 28) -> (batch, 40, 24, 24)
        X = self.conv1(X)
        X = F.relu(X)

        # Apply first pooling layer
        # (batch, 40, 24, 24) -> (batch, 40, 12, 12)
        X = self.pool(X)

        # Pass X through the second convolution layer
        # (batch, 40, 12, 12) -> (batch, 40, 8, 8)
        X = self.conv2(X)
        X = F.relu(X)

        # Apply second pooling layer
        # (batch, 40, 8, 8) -> (batch, 40, 4, 4)
        X = self.pool(X)

        # Flatten [Batch_size, 40, 4, 4] tensor into a flat [Batch_size, 640] tensor
        X = torch.flatten(X, start_dim=1)

        # Pass X through fully connected hidden layer, then apply ReLU
        # 640 features -> 100 neurons
        X = self.fc1(X)
        X = F.relu(X)

        # Pass X through the output layer
        # 100 neurons -> 10 raw digit scores
        fcl = self.fc_out(X)
        return fcl

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()
    
    
