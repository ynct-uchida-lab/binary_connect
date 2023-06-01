# import modules
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

# **********************************************
# main
# **********************************************
def main():
    
    # -------------------------------------
    # モデルの宣言
    # -------------------------------------
    # モデルのインスタンス
    model = networks.BCCls()
    # デバイスの設定
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 誤差関数
    criterion = nn.CrossEntropyLoss()
    # 最適化器
    optimizer = torch.optim.Adam(params=model.parameters())
    # MNISTのデータローダーを取得
    loaders = load_MNIST()
    
    # -------------------------------------
    # モデルの学習
    # -------------------------------------
    # 学習回数
    epochs = 10
    # 学習結果の保存用
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
    }
    # 学習，テスト用ループ
    for epoch in range(epochs):
        # 学習開始
        model.train()
        train_loss = 0
        train_correct = 0
        train_data_num = 0
        for inputs, labels in loaders['train']:
            # 指定したdeviceに変数を変換
            inputs = inputs.to(device)
            labels = labels.to(device)
            # データの整形
            inputs = inputs.view(-1, 28 * 28)
            # ニューラルネットワークへのデータ入力
            outputs = model(inputs)
            # 誤差を算出
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            # 誤差逆伝搬とパラメーターの更新
            optimizer.zero_grad()
            loss.backward()
            model.fc1.weight.grad = model.fc1_bc.weight.grad
            model.fc2.weight.grad = model.fc2_bc.weight.grad
            optimizer.step()
            # 正解数を求める
            outputs = torch.argmax(outputs, dim=1)
            train_correct += torch.sum(outputs == labels).item()
            # 総データ数を求める
            train_data_num += labels.shape[0]
    
        # テストパート
        # 学習のストップ
        model.eval()
        test_loss = 0
        test_correct = 0
        test_data_num = 0
        with torch.no_grad():
            for inputs, labels in loaders['test']:
                # 指定したdeviceに変数を変換
                inputs = inputs.to(device)
                labels = labels.to(device)
                # データの整形
                inputs = inputs.view(-1, 28 * 28)
                # ニューラルネットワークへのデータ入力
                outputs = model(inputs)
                # 誤差を算出
                test_loss += criterion(outputs, labels).item()
                # 正解数を求める
                outputs = torch.argmax(outputs, dim=1)
                test_correct += torch.sum(outputs == labels).item()
                # 総データ数を求める
                test_data_num += labels.shape[0]
                
        # 平均誤差を算出
        train_loss /= train_data_num
        test_loss /= test_data_num
        train_correct /= train_data_num
        test_correct /= test_data_num
        # 1エポックあたりの誤差と正解率を表示
        print('epoch: {} Train loss (avg): {}, Accuracy: {}'.format(epoch + 1, train_loss, train_correct))
        print('epoch: {} Test loss (avg): {}, Accuracy: {}'.format(epoch + 1, test_loss, test_correct))
        # print(model.fc1.weight.data)
        print(model.fc2.weight.data)
        # print(model.fc1_bc.weight.data)
        # print(model.fc2_bc.weight.data)

        # 誤差と正解率を代入
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_correct)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_correct)

    # 結果の出力と描画
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
    
if __name__ == '__main__':
    main()
