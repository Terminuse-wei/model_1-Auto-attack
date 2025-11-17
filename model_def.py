# model_def.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------- 你原来的定义，保留 -------
class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, pool=False, drop=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = F.relu(x, inplace=True)
        x = self.pool(x); x = self.drop(x)
        return x

class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 224 -> 112 -> 56 -> 28 -> 14
        self.feat = nn.Sequential(
            ConvBNReLU(3,   32, pool=True,  drop=0.05),
            ConvBNReLU(32,  64, pool=True,  drop=0.05),
            ConvBNReLU(64, 128, pool=True,  drop=0.10),
            ConvBNReLU(128,256, pool=True,  drop=0.10),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(256, num_classes)
        # 这些属性通常在训练时写入整模型；给默认值便于容错
        self.class_names = ["normal","network_failure"]
        self.normal_idx  = 0
        self.input_size  = 224
        self.normalize   = {"mean":[0.485,0.456,0.406], "std":[0.229,0.224,0.225]}
        self.target_layer = "feat.3"
    def forward(self, x):
        x = self.feat(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ------- 新增：ResNet18Classifier -------
from torchvision import models

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        if pretrained:
            try:
                backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                backbone = models.resnet18(weights=None)
        else:
            backbone = models.resnet18(weights=None)

        # 与训练脚本一致：feat + gap + fc，方便 Grad-CAM 定位到最后一层卷积
        self.feat = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,   # ← Grad-CAM 目标层（feat.7）
        )
        self.gap = backbone.avgpool
        self.fc  = nn.Linear(backbone.fc.in_features, num_classes)

        # 默认元信息（训练保存整模型时会覆盖）
        self.class_names = ["normal","network_failure"]
        self.normal_idx  = 0
        self.input_size  = 224
        self.normalize   = {"mean":[0.485,0.456,0.406], "std":[0.229,0.224,0.225]}
        self.target_layer = "feat.7"

    def forward(self, x):
        x = self.feat(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

__all__ = [
    "ConvBNReLU",
    "SmallCNN",
    "ResNet18Classifier",
]