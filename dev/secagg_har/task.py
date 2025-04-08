"""secagg_har: A Flower with SecAgg+ app."""

import random
from collections import OrderedDict

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import data_loaders

num_inputs, num_outputs, num_labels = data_loaders.get_shapes('HAR')

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.lr = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        return self.lr(torch.flatten(x, start_dim=1))

def make_net(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = Net()
    net.apply(weight_init)
    return net

def weight_init(m):
    """
    Initializes the weights of the layer with random values.
    m: the layer which gets initialized
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=2.24)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int, seed, bias, server_pc, p, batch_size):
    train_loader, test_loader = data_loaders.load_data(seed=seed)

    x, x, each_worker_data_train, each_worker_label_train = data_loaders.assign_data(train_loader,
        bias=bias,
        device=None,
        num_labels=num_labels,
        num_workers=30,
        server_pc=server_pc,
        p=p,
        seed=seed)

    # Sample a minibatch from the single worker's data
    minibatch = np.random.choice(
        list(range(each_worker_data_train[partition_id].shape[0])),
        size=batch_size,
        replace=False
    )

    train_loader = { 'data': each_worker_data_train[partition_id], 'label': each_worker_label_train[partition_id], 'minibatch': minibatch }

    return train_loader, test_loader

def train(net, train_loader, test_loader, epochs, learning_rate, device):
    # Move model to the specified device and set to training mode
    net.to(device)
    
    # Set up optimizer and loss function (using CrossEntropyLoss for classification)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    minibatch = train_loader['minibatch']
    
    # Training loop over epochs and batches
    for epoch in range(epochs):
        net.train()

        net.zero_grad()
        with torch.enable_grad():
            output = net(train_loader['data'][minibatch])
            loss = criterion(output, train_loader['label'][minibatch])
            loss.backward()
        
        optimizer.step()

    return test(net, test_loader, device)

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)  # move model to GPU if available
    
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    # Disable gradient calculation for efficiency
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = criterion(output, target)
            
            # Multiply by the number of samples in this batch to sum the loss over the entire dataset
            total_loss += loss.item() * data.size(0)
            total_correct += output.argmax(dim=1).eq(target).sum().item()
            total_samples += data.size(0)
    
    loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return loss, accuracy
