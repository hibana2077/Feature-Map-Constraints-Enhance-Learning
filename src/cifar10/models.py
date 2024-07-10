import timm
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class DueHeadNet(nn.Module):
    def __init__(self, num_classes=101, base_model="seresnet18"):
        super(DueHeadNet, self).__init__()
        self.model1 = timm.create_model(base_model, num_classes=num_classes)
        self.model2 = timm.create_model(base_model, num_classes=num_classes)
        self.feature_table = {
            "seresnet18": 512,
            "seresnet34": 512,
            "seresnet50": 2048,
            "seresnet101": 2048,
            "seresnet152": 2048
        }
        self.cls = nn.Sequential(
            nn.Linear(self.feature_table[base_model]*2, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feature_maps_list = []
        fe_map_a = self.model1.forward_features(x)
        fe_map_b = self.model2.forward_features(x)
        feature_maps_list.append(fe_map_a)
        feature_maps_list.append(fe_map_b)
        feature_maps = torch.stack(feature_maps_list, dim=1)
        feature_maps = feature_maps.view(feature_maps.size(0), -1)
        logits = self.cls(feature_maps)
        return logits, feature_maps_list
    
if __name__ == '__main__':
    model = DueHeadNet(num_classes=10, base_model="seresnet18")
    x = torch.randn(4, 3, 32, 32)
    (logits,feature_maps) = model(x)
    print(f"Parameter count(M): {sum(p.numel() for p in model.parameters()) / 1e6}")
    print(f"Output shape: {logits.shape}")
    print(f"Feature maps shape: {feature_maps[0].shape}")