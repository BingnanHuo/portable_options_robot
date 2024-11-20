
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.models as models


class PrintLayer(torch.nn.Module):
    # print input. For debugging
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        print(x.shape)
        
        return x

class EfficientNet(nn.Module):
    def __init__(self,
                 num_classes,
                 num_heads):
        super().__init__()
        
        self.model = nn.ModuleList([models.efficientnet_v2_m(weights='IMAGENET1K_V1') 
                                    for _ in range(num_heads)])
        # Freeze the feature extractor
        for idx in range(num_heads):
            self.model[idx].classifier = nn.Sequential(
                nn.Dropout(p=0.1, inplace=True),
                nn.LazyLinear(num_classes),
                )
            for param in self.model[idx].features.parameters():
                param.requires_grad = False
            for param in self.model[idx].classifier.parameters():
                param.requires_grad = True
                
        self.num_heads = num_heads
        self.num_classes = num_classes
    
    def forward(self, x, logits=False):
        pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)
        
        for idx in range(self.num_heads):
            if logits:
                y = self.model[idx](x)
            else:
                y = F.softmax(self.model[idx](x), dim=-1)
            pred[:,idx,:] = y
                
        return pred



