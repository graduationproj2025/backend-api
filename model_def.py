import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BaseFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(BaseFeatureExtractor, self).__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # x: (B * num_images, C, H, W)
        features = self.features(x)  # -> (B * num_images, feature_dim, 1, 1)
        features = F.adaptive_avg_pool2d(features, 1)  # -> (B * num_images, feature_dim, 1, 1)
        features = features.view(features.size(0), -1)  # -> (B * num_images, feature_dim)
        return features

class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, features):
        # features: (B, num_images, feature_dim)
        scores = self.attention(features).squeeze(-1)  # (B, num_images)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, num_images, 1)
        weighted = features * weights  # (B, num_images, feature_dim)
        combined = weighted.sum(dim=1)  # (B, feature_dim)
        return combined, weights

class MultiImageClassifier(nn.Module):
    def __init__(self, num_classes=3, feature_dim=512):
        super(MultiImageClassifier, self).__init__()
        self.feature_extractor = BaseFeatureExtractor(pretrained=False)
        self.feature_reducer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.attention = AttentionModule(256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, images):
        # تأكد إنك تستقبل صورة واحدة فقط بالشكل (B, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

        features = self.feature_extractor(images)  # (B, 512)
        reduced = self.feature_reducer(features)   # (B, 256)
        logits = self.classifier(reduced)          # (B, num_classes)
        return logits