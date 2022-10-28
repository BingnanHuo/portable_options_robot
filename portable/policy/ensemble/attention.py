import torch
import torch.nn as nn
import torch.nn.functional as F

from portable.plot import plot_attention_diversity


class AttentionEmbedding(nn.Module):

    def __init__(self, 
                embedding_size=64, 
                attention_depth=32, 
                num_attention_modules=8, 
                plot_dir=None):
        super(AttentionEmbedding, self).__init__()
        self.num_attention_modules = num_attention_modules
        self.out_dim = embedding_size
        self.attention_depth = attention_depth

        self.conv1 = nn.LazyConv2d(out_channels=self.attention_depth, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(2)

        self.attention_modules = nn.ModuleList(
            [
                nn.Conv2d(in_channels=self.attention_depth, out_channels=self.attention_depth, kernel_size=1, bias=False) 
                for _ in range(self.num_attention_modules)
            ]
        )

        self.conv2 = nn.Conv2d(in_channels=self.attention_depth, out_channels=64, kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.linear = nn.LazyLinear(self.out_dim)

        self.plot_dir = plot_dir

    def spatial_feature_extractor(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        return x

    def global_feature_extractor(self, x):
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        return x

    def compact_global_features(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = F.normalize(x)
        return x

    def forward(self, x, return_attention_mask=False, plot=False):
        spacial_features = self.spatial_feature_extractor(x)
        attentions = [self.attention_modules[i](spacial_features) for i in range(self.num_attention_modules)]

        # normalize attention to between [0, 1]
        for i in range(self.num_attention_modules):
            N, D, H, W = attentions[i].size()
            attention = attentions[i].view(-1, H*W)
            attention_max, _ = attention.max(dim=1, keepdim=True)
            attention_min, _ = attention.min(dim=1, keepdim=True)
            attentions[i] = ((attention - attention_min)/(attention_max-attention_min+1e-8)).view(N, D, H, W)

        global_features = [self.global_feature_extractor(attentions[i] * spacial_features) for i in range(self.num_attention_modules)]
        if plot:
            plot_attention_diversity(global_features, self.num_attention_modules, save_dir=self.plot_dir)

        # normalize attention to between [0, 1]
        for i in range(self.num_attention_modules):
            N, D, H, W = global_features[i].size()
            feat = global_features[i].view(-1, H*W)
            feat_max, _ = feat.max(dim=1, keepdim=True)
            feat_min, _ = feat.min(dim=1, keepdim=True)
            global_features[i] = ((feat - feat_min)/(feat_max-feat_min+1e-8)).view(N, D, H, W)

        # embedding = torch.cat([self.compact_global_features(f).unsqueeze(0) for f in global_features], dim=0)  # (num_modules, N, embedding_size)

        return global_features if not return_attention_mask else (global_features, attentions)