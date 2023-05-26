# import modules
import torch
import torch.nn.functional as f

class MyNet(torch.nn.Module):
    def __init__(self):
        # ネットワークの定義
        super(MyNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 1000)
        self.fc2 = torch.nn.Linear(1000, 10)
        
    # 順伝搬
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        
        return f.log_softmax(x, dim=1)
    