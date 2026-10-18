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

            # Fully connected hidden layer, 640 input features -> 100 neurons
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

            # Fully connected hidden layer, 640 input features -> 100 neurons
            self.fc1 = nn.Linear(in_features=in_features, out_features=100)

            # Output layer: 100 neurons -> 10 digit scores
            self.fc_out = nn.Linear(in_features=100, out_features=10)
            self.forward = self.model_3

        elif mode == 4:
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

            # First fully connected hidden layer, 640 input features -> 100 neurons
            self.fc1 = nn.Linear(in_features=in_features, out_features=100)
            # Second fully connected hidden layer, 100 inputs from fc1 -> 100 neurons
            self.fc2 = nn.Linear(in_features=100, out_features=100)

            # Output layer
            # 100 inputs from fc2 -> 10 digit scores
            self.fc_out = nn.Linear(in_features=100, out_features=10)
            self.forward = self.model_4

        elif mode == 5:
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

            # First fully connected hidden layer, 640 input features -> 1000 neurons
            self.fc1 = nn.Linear(in_features=in_features, out_features=1000)
            # Second fully connected hidden layer, 1000 inputs from fc1 -> 1000 neurons
            self.fc2 = nn.Linear(in_features=1000, out_features=1000)

            # Add dropout with probability 0.5 during training
            self.dropout = nn.Dropout(p=0.5)

            # Output layer
            # 1000 inputs from fc2 -> 10 digit scores
            self.fc_out = nn.Linear(in_features=1000, out_features=10)
            self.forward = self.model_5

        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model made up of 1 fully connected hidden layer
    def model_1(self, X):
        # X is a tensor consisting of: [Batch_Size, Channels, Height, Width]
        # Flatten [batch, 1, 28, 28] tensor into a flat [batch, 784] tensor
        X = torch.flatten(X, start_dim=1)

        # Pass X through fully connected hidden layer 1, then apply sigmoid
        X = torch.sigmoid(self.fc1(X))

        # Pass X through the fully connected output layer to obtain 10 logits
        fcl = self.fc_out(X)
        return  fcl

    # 2 convolutional layers + 1 fully connnected hidden layer
    def model_2(self, X):        
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
        # 100 features -> 10 logits
        fcl = self.fc_out(X)
        return fcl

    # 2 convolutional layers + 1 fully connected layer, with ReLU instead of Sigmoid
    def model_3(self, X):
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
        # 100 features -> 10 logits
        fcl = self.fc_out(X)
        return fcl

    # 2 convolutional layers + 2 fully connected layers, with ReLU
    def model_4(self, X):
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

        # Pass X through first fully connected hidden layer, then apply ReLU
        # 640 features -> 100 neurons
        X = self.fc1(X)
        X = F.relu(X)

        # Pass X through second fully connected hidden layer, then apply ReLU
        # 100 features -> 100 neurons
        X = self.fc2(X)
        X = F.relu(X)

        # Pass X through the output layer
        # 100 features -> 10 logits
        fcl = self.fc_out(X)
        return fcl

    # 2 convolutional layers + 2 1000-neuron fully connected layers with ReLU + dropout
    def model_5(self, X):
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

        # Pass X through first fully connected hidden layer, then apply ReLU
        # 640 features -> 1000 neurons
        X = self.fc1(X)
        X = F.relu(X)
        X = self.dropout(X)

        # Pass X through second fully connected hidden layer, then apply ReLU
        # 1000 features -> 1000 neurons
        X = self.fc2(X)
        X = F.relu(X)
        X = self.dropout(X)

        # Pass X through the output layer
        # 1000 features -> 10 logits
        fcl = self.fc_out(X)
        return fcl