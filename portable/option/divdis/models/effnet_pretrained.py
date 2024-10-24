
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

        custom_efficientnet = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        custom_efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.LazyLinear(num_classes),
            )

        '''self.model = nn.ModuleList([torch.nn.Sequential(
            *list(efficientnet.children())[:-1],

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5, inplace=True),
            nn.LazyLinear(num_classes),
            ) for _ in range(num_heads)])'''
        
        self.model = nn.ModuleList([custom_efficientnet for _ in range(num_heads)])
        # Freeze the feature extractor
        for idx in range(num_heads):
            for param in self.model[idx].features.parameters():
                param.requires_grad = False
            for param in self.model[idx].classifier.parameters():
                param.requires_grad = True
        
        print(self.model)
        
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



