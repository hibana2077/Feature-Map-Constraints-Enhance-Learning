import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
    
    def forward(self, inputs, targets):
        # 計算 cosine 相似度
        cosine_sim = F.cosine_similarity(inputs, targets, dim=1)
        # 1 - cosine_sim 的平均值可以視為損失
        return 1 - cosine_sim.mean()

if __name__ == "__main__":
        
    # 測試用例
    inputs = torch.randn((4,512,1,1), requires_grad=True)
    targets = torch.randn((4,512,1,1))
    loss_function = CosineSimilarityLoss()
    loss = loss_function(inputs, targets)

    print(loss)