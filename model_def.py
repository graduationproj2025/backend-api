import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BaseFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: (B, C, H, W)
        features = self.features(x)  # (B, 512, H', W')
        features = self.pool(features)  # (B, 512, 1, 1)
        return features.view(features.size(0), -1)  # (B, 512)

class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),  # تغيير من ReLU إلى Tanh للأداء الأفضل
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, features):
        # features: (B, num_images, feature_dim)
        scores = self.attention(features)  # (B, num_images, 1)
        weights = F.softmax(scores, dim=1)  # (B, num_images, 1)
        return (features * weights).sum(dim=1), weights  # (B, feature_dim), (B, num_images, 1)

class MultiImageClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.feature_extractor = BaseFeatureExtractor(pretrained)
        self.feature_reducer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(256, num_classes)
        
        # تهيئة الأوزان
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        features = self.feature_extractor(x)  # (B, 512)
        reduced = self.feature_reducer(features)  # (B, 256)
        return self.classifier(reduced)  # (B, num_classes)

    def load_model(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location=device))
        self.to(device)
        self.eval()
        return self