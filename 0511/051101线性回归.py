"""
pytorch 线性回归实战
"""

from operator import mod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.rand(500,1)
y_real = 3.5*x+6.6

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc = nn.Linear(1,1)

    def forward(self,input):
        output = self.fc(input)
        return output
        
model = Network()
print(model)
optimize = torch.optim.Adam(model.parameters(),0.01)
loss_fn = nn.MSELoss()

for i in range(10000):
    optimize.zero_grad()
    y_predict = model(x)
    loss = loss_fn(y_predict,y_real)
    loss.backward()
    optimize.step()
    if i%50==0:
        print("loss:",loss.item(),"w,b:",list(model.parameters()))



