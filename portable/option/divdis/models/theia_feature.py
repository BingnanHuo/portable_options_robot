
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

class TheiaFeature(nn.Module):
    # assume inputs are after theia_model.forward_feature(x).
    def __init__(self):
        super().__init__()
        
        self.backbone = AutoModel.from_pretrained("theaiinstitute/theia-base-patch16-224-cddsv", 
                                      trust_remote_code=True)
        for param in self.backbone.parameters():
            param.requires_grad = False


    def forward(self, x):
        tokens = self.backbone.forward_feature(x)      # (B,197,D)
        #features = tokens.mean(1)                            # (B,D)

        return tokens


