"""pytorchexample: A Flower / PyTorch app."""

from collections import OrderedDict

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

import numpy as np
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
    net.apply(weight_init)

fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int, args):
    if args['dataset'] == 'HAR':
        start_time = time.time()
        train_loader, test_loader = data_loaders.load_data(args['dataset'], args['seed'])

        x, x, each_worker_data_train, each_worker_label_train = data_loaders.assign_data(train_loader,
            bias=args['bias'],
            device=None,
            num_labels=num_labels,
            num_workers=num_partitions,
            server_pc=args['server_pc'],
            p=args['p'],
            dataset=args['dataset'],
            seed=args['seed'])

        # Sample a minibatch from the single worker's data
        minibatch = np.random.choice(
            list(range(each_worker_data_train[partition_id].shape[0])),
            size=args['batch_size'],
            replace=False
        )

        train_loader = { 'data': each_worker_data_train[partition_id], 'label': each_worker_label_train[partition_id], 'minibatch': minibatch }
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds p{partition_id}")

    elif args['dataset'] == 'ANOTHER':
        pass

    else:
        raise NotImplementedError

    return train_loader, test_loader

# def load_data(partition_id: int, num_partitions: int, batch_size: int):
#     """Load partition CIFAR10 data."""
#     # Only initialize `FederatedDataset` once
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         fds = FederatedDataset(
#             dataset="uoft-cs/cifar10",
#             partitioners={"train": partitioner},
#         )
#     partition = fds.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     pytorch_transforms = Compose(
#         [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )

#     def apply_transforms(batch):
#         """Apply transforms to the partition from FederatedDataset."""
#         batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#         return batch

#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(
#         partition_train_test["train"], batch_size=batch_size, shuffle=True
#     )
#     testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
#     return trainloader, testloader

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

    val_loss, val_acc = test(net, test_loader, device)

    results = {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results

# def train(net, trainloader, valloader, epochs, learning_rate, device):
#     """Train the model on the training set."""
#     net.to(device)  # move model to GPU if available
#     criterion = torch.nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
#     net.train()
#     for _ in range(epochs):
#         for batch in trainloader:
#             images = batch["img"]
#             labels = batch["label"]
#             optimizer.zero_grad()
#             criterion(net(images.to(device)), labels.to(device)).backward()
#             optimizer.step()

#     val_loss, val_acc = test(net, valloader, device)

#     results = {
#         "val_loss": val_loss,
#         "val_accuracy": val_acc,
#     }
#     return results

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
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# def test(net, testloader, device):
#     """Validate the model on the test set."""
#     net.to(device)  # move model to GPU if available
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for batch in testloader:
#             images = batch["img"].to(device)
#             labels = batch["label"].to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     loss = loss / len(testloader)
#     return loss, accuracy
