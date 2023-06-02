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
    
class BCCls(torch.nn.Module):
    def set_grad(self):
        self.fc1.weight.grad =  self.fc1_bc.weight.grad
        self.fc2.weight.grad = self.fc2_bc.weight.grad
        
    def binarize(self):
        self.fc1_bc.weight.data = torch.sign(self.fc1.weight.data)
        self.fc2_bc.weight.data = torch.sign(self.fc2.weight.data)
    
    def __init__(self):
        # ネットワークの定義
        super(BCCls, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 1000)
        self.bc = torch.nn.BatchNorm1d(1000)
        self.fc2 = torch.nn.Linear(1000, 10)
        self.fc1_bc = torch.nn.Linear(28 * 28, 1000)
        self.fc2_bc = torch.nn.Linear(1000, 10)
        
    # 順伝搬
    def forward(self, x):
        BCCls.binarize(self)
        x = self.fc1_bc(x)
        x = self.bc(x)
        x = torch.relu(x)
        x = self.fc2_bc(x)
        
        return x