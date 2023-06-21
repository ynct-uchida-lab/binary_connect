# import modules
import torch
import torch.nn as nn

import networks
import functions

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
    loaders = functions.load_MNIST()
    
    # -------------------------------------
    # モデルの学習
    # -------------------------------------
    # 学習回数
    epochs = 5
    # 学習結果の保存用
    history = {
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
    }
    # 学習，テスト用ループ
    for epoch in range(epochs):
        losses = [0, 0]
        corrects = [0, 0]
        data_nums = [0, 0]
        accuracy = [0, 0]
        for i, mode in enumerate(['train', 'test']):
            if mode == 'train':
                # モデルを学習モードにする
                model.train()
            else:
                # モデルを推論モードにする
                model.eval()
            for inputs, labels in loaders[mode]:
                # 指定したdeviceに変数を変換
                inputs = inputs.to(device)
                labels = labels.to(device)
                # データの整形
                inputs = inputs.view(-1, 28 * 28)
                # ニューラルネットワークへのデータ入力
                outputs = model(inputs)
                # 誤差を算出
                loss = criterion(outputs, labels)
                losses[i] += loss.item()
                if mode == 'train':
                    # 誤差逆伝搬とパラメーターの更新
                    optimizer.zero_grad()
                    loss.backward()
                    model.set_grad()
                    optimizer.step()
                # 正解数を求める
                outputs = torch.argmax(outputs, dim=1)
                corrects[i] += torch.sum(outputs == labels).item()
                # 総データ数を求める
                data_nums[i] += labels.shape[0]
            # 平均誤差を算出
            losses[i] /= data_nums[i]
            accuracy[i] = corrects[i] / data_nums[i]
            # 誤差と正解率を代入
            history[mode + '_loss'].append(losses[i])
            history[mode + '_acc'].append(accuracy[i])
            
        # 誤差と正解率を表示
        print('epoch: {} Train loss (avg): {}, Accuracy: {}'.format(epoch + 1, losses[0], accuracy[0]))
        print('epoch: {} Test loss (avg): {}, Accuracy: {}'.format(epoch + 1, losses[1], accuracy[1]))
        
        # グラフ造形用にエポック数を追加
        history['epoch'].append(epoch + 1)
        
        # print(model.fc1.weight.data)
        # print(model.fc2.weight.data)
        # print(model.fc1_bc.weight.data)
        # print(model.fc2_bc.weight.data)
        
    # 正答率,誤差のグラフを出力
    functions.make_fig(history)
    
if __name__ == '__main__':
    main()
