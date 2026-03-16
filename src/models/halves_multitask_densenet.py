from __future__ import annotations

import torch
import torch.nn as nn


class DenseNet121HalvesMultitask(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        try:
            from torchvision.models import densenet121, DenseNet121_Weights
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
            base = densenet121(weights=weights)
        except Exception:
            from torchvision.models import densenet121
            base = densenet121(weights=None)

        self.features = base.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = base.classifier.in_features

        # Side heads
        self.head_left = nn.Linear(feat_dim, 5)
        self.head_right = nn.Linear(feat_dim, 5)

        # Global head uses left, right, and asymmetry
        self.head_bin = nn.Sequential(
            nn.Linear(feat_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 1),
        )

    def encode_branch(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x_left, x_right):
        f_left = self.encode_branch(x_left)
        f_right = self.encode_branch(x_right)

        logits_left = self.head_left(f_left)
        logits_right = self.head_right(f_right)

        f_diff = torch.abs(f_left - f_right)
        f_global = torch.cat([f_left, f_right, f_diff], dim=1)
        logits_bin = self.head_bin(f_global).squeeze(1)

        return {
            "logits_bin": logits_bin,
            "logits_left": logits_left,
            "logits_right": logits_right,
        }


def freeze_backbone(model: DenseNet121HalvesMultitask):
    for p in model.features.parameters():
        p.requires_grad = False

    for p in model.head_bin.parameters():
        p.requires_grad = True
    for p in model.head_left.parameters():
        p.requires_grad = True
    for p in model.head_right.parameters():
        p.requires_grad = True


def unfreeze_last_block(model: DenseNet121HalvesMultitask):
    for p in model.features.parameters():
        p.requires_grad = False

    for name, p in model.features.named_parameters():
        if name.startswith("denseblock4") or name.startswith("norm5"):
            p.requires_grad = True

    for p in model.head_bin.parameters():
        p.requires_grad = True
    for p in model.head_left.parameters():
        p.requires_grad = True
    for p in model.head_right.parameters():
        p.requires_grad = True


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]