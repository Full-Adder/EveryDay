import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize

TRAIN_BATCH_SIZE = 128
TEST_BAICH_SIZE = 128*128
os.chroot = '~/WorkPlace/Python/VScode_Hello/0511'

#准备数据
def get_dataloader(batch_size =TRAIN_BATCH_SIZE,train = True):
    transform_fn = Compose([
        ToTensor(),
        Normalize((0.137,),(0.3081,))
    ])
    data_set = MNIST('./data/',train,transform_fn)
    # data_set[0]=(PIL.Image.Image image mode=L size=28x28), 5)
    data_loader = DataLoader(data_set,batch_size,True)
    return data_loader

#定义网络
class Network(nn.Module):
    def __init__(self) -> None:
        super(Network,self).__init__()
        self.fc1 = nn.Linear(28*28,28)
        self.fc2 = nn.Linear(28,10)

    def forward(self,input):
        """
        input:[batch_size,1,28,28]//[batch,chanl,w,h]
        output:[10]
        """
        input = input.view([input.size(0),1*28*28])
        output = self.fc1(input)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.log_softmax(output,-1)
        return output

#实例化网络、定义优化器和损失函数
model = Network()
optimize = Adam(model.parameters(),0.1)
loss_fn = F.nll_loss

#训练
def train(echo = 1000):
    loss = []
    for i in range(echo):
        train_data_loader = get_dataloader(TRAIN_BATCH_SIZE)
        for id,(input,target) in enumerate(train_data_loader):
            optimize.zero_grad()
            output = model(input)
            cur_loss = loss_fn(output,target)
            cur_loss.backward()
            optimize.step()

            if id % 100 == 0:
                print(i,':',cur_loss.item())
                loss.append(cur_loss.item())
    x = [i for i in range(loss.size())]
    plt.plot(x,loss)
    plt.show()
                
        
def test():
    loss = []
    acc = []
    test_data_loader = get_dataloader(TEST_BAICH_SIZE,False)
    for id,(input,target) in enumerate(test_data_loader):
        with torch.no_grad():
            output = model(input)
            #loss:
            cur_loss = loss_fn(output,target)
            loss.append(cur_loss.item())
            #acc:
            pred = output.max(dim = -1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc.append(cur_acc.item())
    print('loss:',np.mean(loss),'acc:',np.mean(acc))




if __name__ == '__main__':
    train(10)
    test()