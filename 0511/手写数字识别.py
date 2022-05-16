import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

# 准备数据集

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000


def get_dataloader(train=True, batch_size=BATCH_SIZE):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.137,), std=(0.3081,))
    ])

    data_set = MNIST(root='./data/', transform=transform_fn, train=train)
    data_loader = DataLoader(data_set, batch_size, True)
    return data_loader


# 构建模型

class MNIST_Model(nn.Module):
    def __init__(self):
        super(MNIST_Model, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, input):
        """ 
        input : [batch_size,1,28,28]
        """
        x = input.view([input.size(0), 1 * 28 * 28])
        x1 = self.fc1(x)
        x1 = F.relu(x1)
        x2 = self.fc2(x1)
        return F.log_softmax(x2, -1)


model = MNIST_Model()
optimizer = Adam(model.parameters(), 0.01)
if os.path.exists(root + 'model.pkl'):
    model.load_state_dict(torch.load(root + 'model.pkl'))
    optimizer.load_state_dict(torch.load(root + 'optim.pkl'))


def train(echo=1000):
    for i in range(echo):
        data_loader = get_dataloader()
        for id, (input, label) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(input)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()

            if id % 10 == 0:
                print(i, ":", id, loss.item())

            if id % 1000 == 0:
                torch.save(model.state_dict(), root + 'model.pkl')
                torch.save(optimizer.state_dict(), root + 'optim.pkl')


def test():
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(False, TEST_BATCH_SIZE)
    for id, (input, label) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output, label)
            loss_list.append(cur_loss)
            # 计算精确度 output:[batch_size,10] label:[batch_size]
            pred = output.max(dim=1)[-1]
            cur_acc = pred.eq(label).float().mean()
            acc_list.append(cur_acc)
            print('loss:', np.mean(loss_list), '  acc:', np.mean(acc_list))


if __name__ == '__main__':
    test()
