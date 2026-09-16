import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CBAM Sub-module).
    Aggregates spatial information via AvgPool and MaxPool, then applies a shared MLP.
    """
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        mid_planes = max(1, in_planes // reduction)
        self.fc1 = nn.Conv2d(in_planes, mid_planes, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (CBAM Sub-module).
    Aggregates channel information via channel-wise AvgPool and MaxPool,
    followed by a 7x7 spatial convolution.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module combining Channel Attention and Spatial Attention.
    """
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class AttentionMobileNetV3(nn.Module):
    """
    MobileNetV3-Small augmented with a CBAM Attention block prior to the pooling stage.
    Optimized for fine-grained foliar disease feature localization.
    """
    def __init__(self, num_classes=38, use_attention=True, pretrained=True):
        super(AttentionMobileNetV3, self).__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)
        
        self.features = base_model.features
        self.use_attention = use_attention
        
        in_channels = 576
        if self.use_attention:
            self.cbam = CBAM(in_planes=in_channels, reduction=16)
        
        self.avgpool = base_model.avgpool
        
        in_features = base_model.classifier[0].in_features  # 576
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )

        # Unfreeze all layers for fine-tuning
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        if self.use_attention:
            x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNet50Classifier(nn.Module):
    """
    ResNet50 Deep Residual Baseline Classifier.
    """
    def __init__(self, num_classes=38, pretrained=True):
        super(ResNet50Classifier, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Unfreeze all layers for fine-tuning
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)


def build_legacy_mobilenet_v3(num_classes=38):
    """
    Constructs model with legacy classifier structure matching existing saved checkpoints.
    """
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    return model


def build_model(num_classes=38, model_type="attention_mobilenet", pretrained=True):
    """
    Model factory function.
    model_type: 'attention_mobilenet', 'mobilenet_v3', 'resnet50', or 'legacy'
    """
    if model_type == "attention_mobilenet":
        return AttentionMobileNetV3(num_classes=num_classes, use_attention=True, pretrained=pretrained)
    elif model_type == "mobilenet_v3":
        return AttentionMobileNetV3(num_classes=num_classes, use_attention=False, pretrained=pretrained)
    elif model_type == "resnet50":
        return ResNet50Classifier(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "legacy":
        return build_legacy_mobilenet_v3(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
