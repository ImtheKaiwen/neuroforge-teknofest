import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class RadiologyEncoder(nn.Module):
    def __init__(self):
        super(RadiologyEncoder, self).__init__()
        # Batch Normalization eklendi: 3D MRI verilerinde eğitimi inanılmaz stabilize eder.
        self.features = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=3, padding=1), 
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.AdaptiveAvgPool3d(1) # Farklı boyutlardaki MRI'ları sabit boyuta çeker
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)