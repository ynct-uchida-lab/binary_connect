# import modules
import torch

class MLPCls(torch.nn.Module):
    def __init__(self):
        # ネットワークの定義
        super(MLPCls, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 1000)
        self.fc2 = torch.nn.Linear(1000, 10)
        
    # 順伝搬
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        
        return x
    