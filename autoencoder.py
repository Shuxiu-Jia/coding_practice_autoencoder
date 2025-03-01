# Python3
# -*- coding: utf-8 -*-
# Adapted from: lina, https://github.com/Nana0606/autoencoder
# @Author  : Justin
# @Time    : 2025/03/01
# @version : 1.0.0

"""
Autoencoder with single hidden layer using PyTorch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


torch.manual_seed(7701)
np.random.seed(7701)

ENCODING_DIM_INPUT = 784
ENCODING_DIM_OUTPUT = 2
EPOCHS = 10
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.ReLU()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

def train(x_train):
    """
    Build and train autoencoder.
    :param x_train: the train data
    :return: trained autoencoder model
    """
  
    x_train_tensor = torch.FloatTensor(x_train).to(DEVICE)
    
    train_dataset = TensorDataset(x_train_tensor, x_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = Autoencoder(ENCODING_DIM_INPUT, ENCODING_DIM_OUTPUT).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for data, _ in train_loader:
            outputs = model(data)
            loss = criterion(outputs, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}')
    
    return model

def plot_representation(encode_images, y_test):
    """
    Plot the hidden representation.
    :param encode_images: the images after encoding
    :param y_test: the label.
    :return:
    """
    plt.scatter(encode_images[:, 0], encode_images[:, 1], c=y_test, s=3)
    plt.colorbar()
    plt.show()

def show_images(decode_images, x_test):
    """
    Plot the original and reconstructed images.
    :param decode_images: the images after decoding
    :param x_test: testing data
    :return:
    """
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        ax.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        ax.imshow(decode_images[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True,
                                   transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True,
                                  transform=transforms.ToTensor())
    
    x_train = train_dataset.data.float() / 255.0
    y_train = train_dataset.targets
    x_test = test_dataset.data.float() / 255.0
    y_test = test_dataset.targets
    
    x_train = x_train.reshape(x_train.shape[0], -1).numpy()  # Convert to numpy for training
    x_test = x_test.reshape(x_test.shape[0], -1).numpy()
    
    model = train(x_train)
    
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(x_test).to(DEVICE)
        encode_images = model.encode(x_test_tensor).cpu().numpy()
        decode_images = model(x_test_tensor).cpu().numpy()
    
    plot_representation(encode_images, y_test)
    show_images(decode_images, x_test)


