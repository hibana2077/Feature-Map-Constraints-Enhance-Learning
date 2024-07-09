import timm
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class DueHeadNet(nn.Module):
    def __init__(self, num_classes=101):
        super(DueHeadNet, self).__init__()
        self.model1 = timm.create_model("seresnet18", num_classes=num_classes)
        self.model2 = timm.create_model("seresnet18", num_classes=num_classes)
        self.fusion_cls = 

    def forward(self, x):
        feature_maps = []
        return x
    
if __name__ == '__main__':
    model = DueHeadNet()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(f"Parameter count(M): {sum(p.numel() for p in model.parameters()) / 1e6}")
    print(y.size())