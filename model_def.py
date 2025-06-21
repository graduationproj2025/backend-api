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
        batch_size, num_images, c, h, w = x.size()
        x = x.view(-1, c, h, w)  # دمج الصور لتغذية الشبكة دفعة واحدة
        features = self.features(x)
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(batch_size, num_images, -1)
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
        scores = self.attention(features).squeeze(-1)  # (B, num_images)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, num_images, 1)
        weighted = features * weights  # (B, num_images, feature_dim)
        combined = weighted.sum(dim=1)  # (B, feature_dim)
        return combined, weights

class MultiImageClassifier(nn.Module):
    def __init__(self, num_classes=3, num_image_types=2, feature_dim=512):
        super(MultiImageClassifier, self).__init__()
        self.feature_extractor = BaseFeatureExtractor(pretrained=False)
        self.feature_reducer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.attention = AttentionModule(256)
        self.classifier = nn.Linear(256, num_classes)  # ✅ تعديل هنا بدون nn.Sequential

    def forward(self, images):
        features = self.feature_extractor(images)  # (B, num_images, 512)
        batch_size, num_images, _ = features.size()
        features = features.view(batch_size * num_images, -1)
        reduced = self.feature_reducer(features)
        reduced = reduced.view(batch_size, num_images, -1)
        combined, attn_weights = self.attention(reduced)
        logits = self.classifier(combined)
        return logits, attn_weights
