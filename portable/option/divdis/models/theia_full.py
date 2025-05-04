
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.models as models
from transformers import AutoModel, AutoProcessor


class PrintLayer(torch.nn.Module):
    # print input. For debugging
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x

class TheiaFull(nn.Module):
    def __init__(self,
                 num_classes,
                 num_heads):
        super().__init__()
        
        '''self.feature = nn.ModuleList([
            # Theia-CDDSV: CLIP, Dino, Depth-Anything, SAM, ViT
            AutoModel.from_pretrained("theaiinstitute/theia-base-patch16-224-cddsv", 
                                      trust_remote_code=True)
                for _ in range(num_heads)
        ])
        # Freeze feature extraction layer weights
        for idx in range(num_heads):
            for param in self.feature[idx].parameters():
                param.requires_grad = False'''
        # Theia-CDDSV: CLIP, Dino, Depth-Anything, SAM, ViT
        self.backbone = AutoModel.from_pretrained("theaiinstitute/theia-base-patch16-224-cddsv", 
                                      trust_remote_code=True)
        # Freeze feature extraction layer weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        print(self.backbone.config)
        hidden_dim = 768 #self.backbone.config.hidden_size   # 768 for 'base'
        
        '''self.classifier = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(output_size=1),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.LayerNorm(normalized_shape=(768,), eps=1e-05, elementwise_affine=True),
                nn.LazyLinear(out_features=750, bias=True),
                nn.Tanh(),
                nn.LazyLinear(out_features=2, bias=False)
            ) for _ in range(num_heads)
        ])'''
        self.classifier = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)               # logits
            ) for _ in range(num_heads)
        ])
        
        self.num_heads = num_heads
        self.num_classes = num_classes

    
    def forward(self, x, logits=False):
        pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)
        
        '''for idx in range(self.num_heads):
            if logits:
                z = self.feature.forward_feature(x)
                z = z.transpose(1,2)
                y = self.classifier[idx](z)
            else:
                z = self.feature.forward_feature(x)
                z = z.transpose(1,2)
                y = F.softmax(self.classifier[idx](z), dim=-1)
            pred[:,idx,:] = y'''

        for idx in range(self.num_heads):
            tokens = self.backbone.forward_feature(x)      # (B,197,D)
            features = tokens.mean(1)                            # (B,D)
            out = self.classifier[idx](features)                # (B,2)

            if not logits:                                     # probas
                out = F.softmax(out, dim=-1)

            pred[:, idx, :] = out
        return pred



