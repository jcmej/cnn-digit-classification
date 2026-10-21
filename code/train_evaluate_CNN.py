from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ConvNet import ConvNet 
import argparse
import numpy as np 
import matplotlib.pyplot as plt

def train(model, device, train_loader, optimizer, criterion):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda', 'mps', or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # Compute loss based on criterion
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        
        # Count correct predictions overall
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    train_loss = float(np.mean(losses))
    train_acc = 100 * correct / len(train_loader.dataset)
    print(f'  Training | loss: {train_loss:.4f} | accuracy: {train_acc:.2f}% '
          f'({correct}/{len(train_loader.dataset)} correct)', flush=True)
    return train_loss, train_acc
    


def test(model, device, test_loader, criterion):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    criterion: used to compute loss for prediction and target
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            

            # Predict for data by doing forward pass
            output = model(data)
            
            # Compute loss based on same criterion as training 
            loss = criterion(output, target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            
            # Count correct predictions overall 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'  Test     | loss: {test_loss:.4f} | accuracy: {accuracy:.2f}% '
          f'({correct}/{len(test_loader.dataset)} correct)', flush=True)
    
    return test_loss, accuracy

def make_plots(epochs, history, mode):
    os.makedirs('plots', exist_ok=True)
    # Plot training and test loss
    plt.figure()
    plt.plot(epochs, history['train_loss'], marker='o', label='Training')
    plt.plot(epochs, history['test_loss'], marker='o', label='Test')
    plt.xlabel('Epoch')
    if len(epochs) == 1:
        plt.xticks([epochs[0]])
    plt.ylabel('Loss')
    plt.title(f'Mode {mode}: Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/mode_{mode}_loss.png')
    plt.close()
    # Plot training and test accuracy
    plt.figure()
    plt.plot(epochs, history['train_accuracy'], marker='o', label='Training')
    plt.plot(epochs, history['test_accuracy'], marker='o', label='Test')
    plt.xlabel('Epoch')
    if len(epochs) == 1:
        plt.xticks([epochs[0]])
    plt.ylabel('Accuracy (%)')
    plt.title(f'Mode {mode}: Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/mode_{mode}_accuracy.png')
    plt.close()

def run_main(FLAGS):
    # Check if cuda is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'\nMode {FLAGS.mode} | device: {device} | optimizer: SGD | '
          f'learning rate: {FLAGS.learning_rate} | epochs: {FLAGS.num_epochs} | '
          f'batch size: {FLAGS.batch_size}', flush=True)
    
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    # Define optimizer function
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=FLAGS.learning_rate,
        momentum=0.0,
        weight_decay=0.0
    )
        
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    print('Loading MNIST dataset files', flush=True)
    
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4)
    
    best_accuracy = 0.0
    best_epoch = 0

    history = {
        'train_loss': [],
        'test_loss': [],
        'train_accuracy': [],
        'test_accuracy': []
    }
    
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        print(f'\nEpoch {epoch}/{FLAGS.num_epochs}', flush=True)
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_accuracy'].append(train_accuracy)
        history['test_accuracy'].append(test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
    
    print(f'\nBest test accuracy: {best_accuracy:.2f}% (epoch {best_epoch})')
    make_plots(range(1, FLAGS.num_epochs +1), history, FLAGS.mode)
    print('Training and evaluation finished. Plots saved')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=60,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    
    