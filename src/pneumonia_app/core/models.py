"""ResNet50 + ConditionalResNet50 matching training architecture."""
import torch
import torch.nn as nn
from torchvision import models


class ConditionalResNet50(nn.Module):
    def __init__(self, cond_dim, dropout=0.3):
        super().__init__()
        base = models.resnet50(weights=None)
        self.num_ftrs = base.fc.in_features
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.cond_projection = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Linear(128, 512),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.num_ftrs + 512, 256),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, img, cond):
        f = self.backbone(img).view(img.size(0), -1)
        c = self.cond_projection(cond)
        return self.classifier(torch.cat([f, c], dim=1))


def build_model(is_conditional, cond_dim=22):
    if is_conditional:
        return ConditionalResNet50(cond_dim=cond_dim)
    m = models.resnet50(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 1))
    return m


def load_model(path, is_conditional, cond_dim, device):
    model = build_model(is_conditional, cond_dim)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def parse_experiment(name):
    is_cond = name.startswith("conditional_")
    if "masked_roi" in name: mode = "masked_roi"
    elif "roi" in name:      mode = "roi"
    else:                    mode = "plain"
    return is_cond, mode


def get_target_layer(model, is_conditional):
    return model.backbone[-2] if is_conditional else model.layer4