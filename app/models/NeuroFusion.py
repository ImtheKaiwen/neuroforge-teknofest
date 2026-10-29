import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from .Pathology import PathologyEncoder
from .Radiology import RadiologyEncoder

class NeuroFusionModel(nn.Module):
    def __init__(self, num_classes=3):
        super(NeuroFusionModel, self).__init__()
        self.rad_encoder = RadiologyEncoder()
        self.path_encoder = PathologyEncoder()
        
        # Fusion Katmanı (Daha derin MLP)
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes) # Örn: IDH mutasyonu, MGMT, Sağkalım Sınıfı
        )

    def forward(self, mri_data, slide_patches):
        rad_feats = self.rad_encoder(mri_data)
        path_feats = self.path_encoder(slide_patches)
        
        combined = torch.cat((rad_feats, path_feats), dim=1)
        return self.fusion_layer(combined)
