import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000


# 准备数据
def get_dataloader(batch_size=TRAIN_BATCH_SIZE, train=True):
    transform_fn = Compose([
        ToTensor(),
        Normalize((0.137,), (0.3081,))
    ])
    data_set = MNIST('../data/', train, transform_fn)
    # data_set[0]=(PIL.Image.Image image mode=L size=28x28), 5)
    data_loader = DataLoader(data_set, batch_size, True)
    return data_loader


# 定义网络
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, input):
        """
        input:[batch_size,1,28,28]//[batch,chanel ,w,h]
        output:[10]
        """
        input = input.view([input.size(0), 1 * 28 * 28])
        output = self.fc1(input)
        output = F.relu(output)
        output = self.fc2(output)
        return F.log_softmax(output, -1)


# 实例化网络、定义优化器和损失函数
model = Network()
optimize = Adam(model.parameters(), 0.001)
loss_fn = F.nll_loss

if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load('./model/model.pkl'))
    optimize.load_state_dict(torch.load('./model/optim.pkl'))


# 训练
def train(echo):
    loss = []
    acc = []
    for i in range(echo):
        train_data_loader = get_dataloader(TRAIN_BATCH_SIZE)
        model.eval()
        for id, (input, target) in enumerate(train_data_loader):
            optimize.zero_grad()
            output = model(input)
            cur_loss = loss_fn(output, target)
            cur_loss.backward()
            optimize.step()

            if id % 10 == 0:
                print('echo:', i, 'id:', id, 'loss:', cur_loss.item())
                loss.append(cur_loss.item())
                acc.append(test())

            if id % 100 == 0:
                torch.save(model.state_dict(), './model/model.pkl')
                torch.save(optimize.state_dict(), './model/optim.pkl')

    loss_x = [i for i in range(len(loss))]
    acc_x = np.linspace(0, len(loss), len(acc))
    plt.title('loss and acc')
    plt.plot(loss_x, loss, label='loss')
    plt.plot(acc_x, acc, 10, label='acc', color='y')
    plt.ylim(0, 3)
    plt.legend()
    plt.savefig('lossANDacc.svg')
    plt.show()


def test():
    loss = []
    acc = []
    test_data_loader = get_dataloader(TEST_BATCH_SIZE, False)
    for id, (input, target) in enumerate(test_data_loader):
        with torch.no_grad():
            output = model(input)
            # loss:
            cur_loss = loss_fn(output, target)
            loss.append(cur_loss.item())
            # acc:
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc.append(cur_acc.item())
    print('loss:', np.mean(loss), 'acc:', np.mean(acc))
    return np.mean(acc)


if __name__ == '__main__':
    train(3)
