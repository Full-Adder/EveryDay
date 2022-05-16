import torch

device = torch.device('cuda')
x = torch.rand(1,1).to(device)
print(x)
a = []
a.append(x.item())
print(a)