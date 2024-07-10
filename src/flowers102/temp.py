import timm
import torch

test_tensor = torch.randn(4, 3, 224, 224)
model_name_list = ["seresnet18", "seresnet34", "seresnet50", "seresnet101", "seresnet152"]
for model_name in model_name_list:
    model = timm.create_model(model_name, num_classes=10)
    logits = model.forward_features(test_tensor)
    print(f"Model: {model_name}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameter count(M): {sum(p.numel() for p in model.parameters()) / 1e6}")