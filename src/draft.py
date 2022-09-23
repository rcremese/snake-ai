import torch

t = torch.eye(3).repeat(10,1,1)
print(t)
print(t.view(-1, 9))