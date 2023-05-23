
import torch 
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import os

def load_dataframe(file_path):
    dataframe = pd.read_excel(file_path)
    names = dataframe['Name']
    values = dataframe['Overall']
    return names, values

def load_image(image_path, size):
    if os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize(size)
        data = np.asarray(image) / 255.0
        return data
    return None

def create_dataset(names, values, image_dir, size, mean, std):
    X = []
    Y = []
    for i in range(len(names)):
        image_path = os.path.join(image_dir, names[i])
        label = values[i]
        data = load_image(image_path, size)
        if data is not None:
            data = (data - mean) / std
            data = np.transpose(data, (2, 0, 1))
            X.append(data)
            Y.append([float(label)])
    return np.array(X), np.array(Y)

def create_dataloader(X, Y, batch_size):
    imgs_tensor = torch.FloatTensor(X)
    dataloader = DataLoader(imgs_tensor, batch_size=batch_size, shuffle=False)
    return dataloader, Y

def loadData():
    path_to_train = "train/"
    path_to_annotations = "train/annotations.xlsx"
    image_dir = os.path.join(path_to_train, "images")
    size = (224, 224)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    batch_size = 32

    names, values = load_dataframe(path_to_annotations)
    X, Y = create_dataset(names, values, image_dir, size, mean, std)
    dataloader, Y = create_dataloader(X, Y, batch_size)
    return dataloader, Y

dataloader, Y = loadData()
# print(Y)


from torchvision.models import vgg19
from torchvision.transforms import ToTensor, Resize

# Define the custom dataset for Suturing Image Dataset
batch_size = 32

base_model = vgg19(pretrained=True)
base_model.classifier[6] = nn.Linear(4096, 1)

# Freeze the weights of the convolutional layers
for param in base_model.features.parameters():
    param.requires_grad = False

# Define the custom regression model
model = base_model

# Define the optimizer, loss function, and device
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

loss_fn = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def graphLoss(TrainLoss,TestLoss,Epoch):
  x = Epoch
  plt.plot(x,TrainLoss,color='green', label = 'Training Loss') 
  plt.plot(x,TestLoss,color='blue',label= 'Testing Loss')
  plt.legend()
  plt.show()

def graphLoss2(TrainLoss,TestLoss,Epoch):
  x = Epoch
  plt.plot(x,TrainLoss,color='red', label = 'Training Loss') 
  plt.plot(x,TestLoss,color='purple',label= 'Testing Loss')
  plt.legend()
  plt.show()

# def compute_test_loss(dataloader, model, Y, criterion, device,b):
#     loss = 0
#     test_batch = 0

#     for i, inputs in enumerate(dataloader):
#         if i == b:
#             batch_size = len(inputs)
#             images = inputs.to(device)
#             labels = torch.FloatTensor(np.asarray(Y[i * batch_size: min((i + 1) * batch_size, len(Y))]))
#             labels = labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             test_batch += 1

#     return loss / test_batch


def initialize_model():
    base_model = vgg19(pretrained=True)
    base_model.classifier[6] = nn.Linear(4096, 1)

    # Freeze the weights of the convolutional layers
    for param in base_model.features.parameters():
        param.requires_grad = False

    return base_model

def train_epoch(model, data, labels, optimizer, criterion, device):
    running_loss = 0.0
    num_batches = 0

    for i, inputs in enumerate(data):
        num_batches += 1
        images = inputs.to(device)
        batch_size = len(images)
        labels_batch = torch.FloatTensor(np.asarray(labels[i * batch_size: min((i + 1) * batch_size, len(labels))]))
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / num_batches

def compute_test_loss(data_loader, model, labels, criterion, device):
    test_loss = 0.0
    test_batches = 0

    for i, inputs in enumerate(data_loader):
        test_batches += 1
        images = inputs.to(device)
        labels_batch = torch.FloatTensor(np.asarray(labels[i * batch_size: min((i + 1) * batch_size, len(labels))]))
        labels_batch = labels_batch.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels_batch)

        test_loss += loss.item()

    return test_loss / test_batches


class Model:
    def __init__(self, epochs, lr, loss_fn):
        self.epochs = epochs
        self.learning_rate = lr
        self.loss_fn = loss_fn
        self.model = None
        self.train_loss = []
        self.test_loss = []
        self.epoch_list = []



    def train(self, to_train, Y):
        train_losses = []
        test_losses = []
        epoch_list = []

        batch_size = 32

        model = initialize_model()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = self.loss_fn
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            train_loss = train_epoch(model, to_train, Y, optimizer, criterion, device)
            test_loss = float(compute_test_loss(dataloader, model, Y, criterion, device))

            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}")

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            epoch_list.append(epoch + 1)

        self.epoch_list = epoch_list
        self.test_loss = test_losses
        self.train_loss = train_losses
        self.model = model



# Example usage
learning_rate = 0.001
loss_fn = nn.MSELoss()

M = Model(10, learning_rate, loss_fn)
M.train(dataloader, Y)
