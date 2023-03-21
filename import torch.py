import torch

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.mean(dim=0))