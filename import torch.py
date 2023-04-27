import torch

#x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
#print(torch.ones_like(x))
#print(torch.zeros_like(x))

x = torch.FloatTensor([[1, 2], [3, 4]])

print(x.mul(2.))
x=x.mul(2.)
print(x.mul_(3))
print(x) 