import torch
import torch.nn as nn
import torch.nn.functional as F

inputs = torch.randn((4,512,1,1), requires_grad=True)
targets = torch.randn((4,512,1,1))

print(F.cosine_similarity(inputs, targets, dim=1))
print(F.cosine_similarity(inputs, targets, dim=1).mean())

print(F.cosine_similarity(inputs, inputs, dim=1))
print(F.cosine_similarity(inputs, inputs, dim=1).mean())