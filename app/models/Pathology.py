import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class PathologyEncoder(nn.Module):
    def __init__(self):
        super(PathologyEncoder, self).__init__()
        # PyTorch'un güncel ağırlık çağırma yöntemi kullanıldı
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Gated Attention Mekanizması (Daha hassas patch ağırlıklandırması)
        self.attention_V = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(128, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # x shape: (Batch, N_patches, 3, 224, 224)
        # Sıkıştırma (Squeeze) işlemi batch boyutunu kaybetmemek için dikkatli yapılmalı
        b, n, c, h, w = x.shape
        x = x.view(-1, c, h, w) # Patchleri tek boyuta düzle
        
        feats = self.feature_extractor(x)
        feats = feats.view(b, n, -1) # (B, N, 512)
        
        # Gated Attention Hesaplaması
        A_V = self.attention_V(feats)
        A_U = self.attention_U(feats)
        A = self.attention_weights(A_V * A_U) # (B, N, 1)
        A = F.softmax(A, dim=1)
        
        # Agregasyon
        weighted_feats = torch.sum(A * feats, dim=1) # (B, 512)
        return self.fc(weighted_feats)
