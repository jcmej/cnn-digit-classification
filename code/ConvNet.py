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
            self.fc1 = nn.Linear(in_features=in_features, out_features=100)
            self.fc_out = nn.Linear(in_features=100, out_features=10)
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
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
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()

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
    
    
