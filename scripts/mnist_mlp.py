# import modules
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import networks

# NNISTをロードする関数
def load_MNIST(batch=128, intensity=1.0):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                        train = True,
                        download = True,
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x * intensity)
                        ])),
        batch_size = batch,
        shuffle = True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                        train = False,
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x * intensity)
                        ])),
        batch_size = batch,
        shuffle = True)
    
    return{'train': train_loader, 'test': test_loader}

# **********************************************
# main
# **********************************************
def main():
    # 学習回数
    epochs = 20
    
    # 学習結果の保存用
    history = {
        'train_loss':[],
        'test_loss':[],
        'test_acc':[],
    }
    
    # ネットワークを構築
    net: torch.nn.Module = networks.MyNet()
    
    # MNISTのデータローダーを取得
    loaders = load_MNIST()
    
    # -------------------------------------
    # 最適化
    # -------------------------------------
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    
    # -------------------------------------
    # モデルの学習
    # -------------------------------------
    
    # 学習，テスト用ループ
    for epoch in range(epochs):
        
        # 学習パート
        loss = None
        # 学習開始
        net.train(True)
        for i,(data, target) in enumerate(loaders['train']):
            # 学習部分
            # 全結合のみのネットワークでは入力を1次元に
            # print(data.shape)
            # torch.Size([128, 1, 28, 28])
            data = data.view(-1, 28*28)
            # print(data.shape)
            # torch.Size([128, 784])
            
            optimizer.zero_grad()
            output = net(data)
            loss = f.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print('Training log: {} epoch ({} / 60000 train. data). Loss: {}'.format(epoch+1,
                                                                                         (i+1)*128,
                                                                                         loss.item())
                      )
        
                
        history['train_loss'].append(loss.detach().numpy())
    
        # テストパート
        # 学習のストップ
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loaders['test']:
                # テスト部分
                data = data.view(-1, 28 * 28)
                output = net(data)
                test_loss += f.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
 
        test_loss /= 10000

        print('Test loss (avg): {}, Accuracy: {}'.format(test_loss,
                                                        correct / 10000))

        history['test_loss'].append(test_loss)
        history['test_acc'].append(correct / 10000)

    # 結果の出力と描画
    print(history)
    plt.figure()
    plt.plot(range(1, epochs + 1), history['train_loss'], label='train_loss')
    plt.plot(range(1, epochs + 1), history['test_loss'], label='test_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('../outputs/loss.png')
 
    plt.figure()
    plt.plot(range(1, epochs + 1), history['test_acc'])
    plt.title('test accuracy')
    plt.xlabel('epoch')
    plt.savefig('../outputs/test_acc.png')
    
if __name__ == '__main__':
    main()
