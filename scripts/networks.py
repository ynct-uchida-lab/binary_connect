# import modules
import torch
import copy

class MLPCls(torch.nn.Module):
    def __init__(self):
        # ネットワークの定義
        super(MLPCls, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 1000)
        self.bn = torch.nn.BatchNorm1d(1000)
        self.fc2 = torch.nn.Linear(1000, 10)
        
    # 順伝搬
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.fc2(x)
        
        return x
    
class BCCls(torch.nn.Module):
    def __init__(self):
        # ネットワークの定義
        super(BCCls, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 1000)
        self.bn = torch.nn.BatchNorm1d(1000)
        self.fc2 = torch.nn.Linear(1000, 10)
        self.fc1_bc = copy.deepcopy(self.fc1)
        self.fc2_bc = copy.deepcopy(self.fc2)
        
    # 順伝搬
    def forward(self, x):
        BCCls.binarize(self)
        x = self.fc1_bc(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.fc2_bc(x)
        
        return x
    
    # 重み更新用に勾配を更新
    def set_grad(self):
        self.fc1.weight.grad = self.fc1_bc.weight.grad
        self.fc2.weight.grad = self.fc2_bc.weight.grad
        
    # 重みを2値化
    def binarize(self):
        self.fc1_bc.weight.data = torch.sign(self.fc1.weight.data)
        self.fc2_bc.weight.data = torch.sign(self.fc2.weight.data)