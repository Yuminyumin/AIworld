import torch

x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

#dim = 0 은 첫번째 차원을 늘리라는 의미
print(torch.cat([x, y], dim=0))