# # import modules
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import networks
# NNISTをロードする関数
def load_MNIST(batch=5000, intensity=1.0):
    train_loader = DataLoader(
        datasets.MNIST(
            '../data/',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * intensity)
            ])),
        batch_size=batch,
        shuffle=True)

    test_loader = DataLoader(
        datasets.MNIST(
            '../data/',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * intensity)
            ])),
        batch_size=batch)
    
    return {'train': train_loader, 'test': test_loader}

# 結果の出力と描画
def makefig(epochs, history):
    plt.figure()
    plt.plot(range(1, epochs + 1), history['train_loss'], label='train_loss')
    plt.plot(range(1, epochs + 1), history['test_loss'], label='test_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('../outputs/loss_bc.png')
    
    plt.figure()
    plt.plot(range(1, epochs + 1), history['train_acc'], label='train_acc')
    plt.plot(range(1, epochs + 1), history['test_acc'], label='test_acc')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('../outputs/acc_bc.png')