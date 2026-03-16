from __future__ import annotations

import torch
import torch.nn as nn


def _make_densenet_encoder(pretrained: bool = True):
    try:
        from torchvision.models import densenet121, DenseNet121_Weights
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        base = densenet121(weights=weights)
    except Exception:
        from torchvision.models import densenet121
        base = densenet121(weights=None)

    features = base.features
    feat_dim = base.classifier.in_features
    return features, feat_dim


class DenseNet121HalvesBinaryTwoEnc(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.features_left, feat_dim_left = _make_densenet_encoder(pretrained=pretrained)
        self.features_right, feat_dim_right = _make_densenet_encoder(pretrained=pretrained)

        if feat_dim_left != feat_dim_right:
            raise ValueError("Left and right encoders have different feature dimensions.")

        self.feat_dim = feat_dim_left
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 1),
        )

    def encode_left(self, x):
        x = self.features_left(x)
        x = torch.relu(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x

    def encode_right(self, x):
        x = self.features_right(x)
        x = torch.relu(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x_left, x_right):
        f_left = self.encode_left(x_left)
        f_right = self.encode_right(x_right)

        f_diff = torch.abs(f_left - f_right)
        f_global = torch.cat([f_left, f_right, f_diff], dim=1)

        logits = self.classifier(f_global).squeeze(1)
        return logits


def freeze_backbone(model: DenseNet121HalvesBinaryTwoEnc):
    for p in model.features_left.parameters():
        p.requires_grad = False
    for p in model.features_right.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_last_block(model: DenseNet121HalvesBinaryTwoEnc):
    for p in model.features_left.parameters():
        p.requires_grad = False
    for p in model.features_right.parameters():
        p.requires_grad = False

    for name, p in model.features_left.named_parameters():
        if name.startswith("denseblock4") or name.startswith("norm5"):
            p.requires_grad = True

    for name, p in model.features_right.named_parameters():
        if name.startswith("denseblock4") or name.startswith("norm5"):
            p.requires_grad = True

    for p in model.classifier.parameters():
        p.requires_grad = True


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]