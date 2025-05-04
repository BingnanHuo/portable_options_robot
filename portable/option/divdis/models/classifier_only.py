
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


class AttentionPool(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.mha   = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, tokens):
        # tokens: (B,197,768)
        B, N, D = tokens.shape
        q = self.query.expand(B, -1, -1)      # (B,1,D)
        out, _ = self.mha(q, tokens, tokens)  # attends query→tokens
        return out.squeeze(1)                 # (B,D)

    
class ClassifierOnly(nn.Module):
    # assume inputs are after theia_model.forward_feature(x).
    def __init__(self,
                 num_classes,
                 num_heads):
        super().__init__()

        self.pool = AttentionPool(768, num_heads=8)

        self.classifier = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)               # logits
            ) for _ in range(num_heads)
        ])
        
        self.num_heads = num_heads
        self.num_classes = num_classes

    
    def forward(self, z, logits=False):
        pred = torch.zeros(z.shape[0], self.num_heads, self.num_classes).to(z.device)
        
        for idx in range(self.num_heads):
            #tokens = self.backbone.forward_feature(x)      # (B,197,D)
            #features = tokens.mean(1)                        # (B,D)

            #features = z                
            #out = self.classifier[idx](features)                # (B,2)

            tokens = z                           # (B,197,768)
            features = self.pool(tokens)                # (B,D)
            out = self.classifier[idx](features)                # (B,2)

            if not logits:                                     # probas
                out = F.softmax(out, dim=-1)

            pred[:, idx, :] = out
        return pred



