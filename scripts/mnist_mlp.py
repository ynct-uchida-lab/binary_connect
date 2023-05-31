# import modules
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import networks

# NNISTをロードする関数
def load_MNIST(batch=128, intensity=1.0):
    train_loader = DataLoader(
        datasets.MNIST('../data/',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * intensity)
            ])),
        batch_size=batch,
        shuffle=True)

    test_loader = DataLoader(
        datasets.MNIST('../data/',
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
    model = networks.MLPCls()
    # デバイスの設定
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 誤差関数
    criterion = nn.CrossEntropyLoss()
    #最適化器
    optimizer = torch.optim.Adam(params=model.parameters())
    # MNISTのデータローダーを取得
    loaders = load_MNIST()
    
    # -------------------------------------
    # モデルの学習
    # -------------------------------------
    # 学習回数
    epochs = 5
    # 学習結果の保存用
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
    }
    # 学習，テスト用ループ
    for epoch in range(epochs):
        # 学習開始
        model.train()
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
            # 誤差逆伝搬とパラメーターの更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # テストパート
        # 学習のストップ
        model.eval()
        test_loss = 0
        correct = 0
        data_num = 0
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
                correct += torch.sum(outputs == labels).item()
                # 総データ数を求める
                data_num += labels.shape[0]
                
        # 平均誤差を算出
        test_loss /= data_num
        correct /= data_num
        # 1エポックあたりの誤差と正解率を表示
        print('epoch{} Test loss (avg): {}, Accuracy: {}'.format(epoch+1, test_loss, correct))

        # 誤差と正解率を代入
        history['train_loss'].append(loss.item())
        history['test_loss'].append(test_loss)
        history['test_acc'].append(correct)

    # 結果の出力と描画
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
