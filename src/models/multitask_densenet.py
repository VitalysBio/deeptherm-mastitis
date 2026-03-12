from __future__ import annotations

import torch
import torch.nn as nn


class DenseNet121Multitask(nn.Module):
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
        in_features = base.classifier.in_features

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Three heads
        self.head_bin = nn.Linear(in_features, 1)
        self.head_left = nn.Linear(in_features, 5)
        self.head_right = nn.Linear(in_features, 5)

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        out_bin = self.head_bin(x).squeeze(1)   # [B]
        out_left = self.head_left(x)            # [B, 5]
        out_right = self.head_right(x)          # [B, 5]

        return {
            "logits_bin": out_bin,
            "logits_left": out_left,
            "logits_right": out_right,
        }


def freeze_backbone(model: DenseNet121Multitask):
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.head_bin.parameters():
        p.requires_grad = True
    for p in model.head_left.parameters():
        p.requires_grad = True
    for p in model.head_right.parameters():
        p.requires_grad = True


def unfreeze_last_block(model: DenseNet121Multitask):
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