import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


def convrelu(in_channels, out_channels, kernel, padding):
    """
    Create a sequential layer of 2D convolution followed by ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ResNet50UNet(nn.Module):
    def __init__(self, n_class):

        super().__init__()
        self.base_model = models.resnet50(weights=None)
        self.base_layers = list(self.base_model.children())

        # Encoder
        self.layer0 = nn.Sequential(*self.base_layers[:3])   # (3→64)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)

        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # (64→256)
        self.layer1_1x1 = convrelu(256, 256, 1, 0)

        self.layer2 = self.base_layers[5]  # (256→512)
        self.layer2_1x1 = convrelu(512, 512, 1, 0)

        self.layer3 = self.base_layers[6]  # (512→1024)
        self.layer3_1x1 = convrelu(1024, 1024, 1, 0)

        self.layer4 = self.base_layers[7]  # (1024→2048)
        self.layer4_1x1 = convrelu(2048, 2048, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Attention Gate
        self.attention_gate4 = AttentionGate(2048, 1024, 512)
        self.attention_gate3 = AttentionGate(1024, 512, 256)
        self.attention_gate2 = AttentionGate(512, 256, 128)
        self.attention_gate1 = AttentionGate(256, 64, 32)

        # Decoder
        self.conv_up3 = convrelu(1024 + 2048, 1024, 3, 1)
        self.conv_up2 = convrelu(512 + 1024, 512, 3, 1)
        self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.dropout = nn.Dropout(p=0.25)
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, layer_input):
        x_original = self.conv_original_size0(layer_input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(layer_input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        layer3_att = self.attention_gate4(g=x, x=layer3)
        x = torch.cat([x, layer3_att], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        layer2_att = self.attention_gate3(g=x, x=layer2)
        x = torch.cat([x, layer2_att], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        layer1_att = self.attention_gate2(g=x, x=layer1)
        x = torch.cat([x, layer1_att], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        layer0_att = self.attention_gate1(g=x, x=layer0)
        x = torch.cat([x, layer0_att], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        x = self.dropout(x)
        out = self.conv_last(x)

        return out

def dice_loss(pred, target, smooth=1.):
    """
    Compute dice loss
    """
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) /
            (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def focal_tversky_loss(pred, target, alpha=0.3, beta=0.7, gamma=1.5, smooth=1.):
    """
    Compute Focal Tversky Loss
    """
    pred = pred.contiguous()
    target = target.contiguous()
    
    TP = (pred * target).sum(dim=2).sum(dim=2)
    FP = (pred * (1 - target)).sum(dim=2).sum(dim=2)
    FN = ((1 - pred) * target).sum(dim=2).sum(dim=2)
    
    tversky_index = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    focal_tversky = torch.pow(1 - tversky_index, gamma)
    
    return focal_tversky.mean()

def laplacian_loss(pred, target, smooth=1e-5):
    """
    Compute Laplacian Loss
    """
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=pred.dtype, device=pred.device).unsqueeze(0).unsqueeze(0)
    
    pred_laplacian = F.conv2d(pred, laplacian_kernel, padding=1)
    target_laplacian = F.conv2d(target, laplacian_kernel, padding=1)
    
    loss = torch.abs(pred_laplacian - target_laplacian).mean()
    
    return loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50UNet(n_class=1)
    model = model.to(device)
    print(model)