import torch

matrix = torch.randn(1000, 1000).cuda()
transposed_matrix = matrix.t()
print(transposed_matrix.shape)