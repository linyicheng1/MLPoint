import torch
from torch import nn
from typing import Optional, Callable
import torch.nn.functional as F
from torch.nn import Module
from torchvision.models import resnet


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> Module:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


class ML_Point(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.ReLU(inplace=True)
        # first layer
        self.res1 = ResBlock(3, 8, 1, downsample=nn.Conv2d(3, 8, 1), gate=self.gate)
        self.conv1 = resnet.conv1x1(8, 16)
        self.conv_head1 = resnet.conv1x1(16, 4)
        # second layer
        self.res2 = ResBlock(8, 16, stride=2, downsample=nn.Conv2d(8, 16, 1, stride=2), gate=self.gate)
        self.conv2 = resnet.conv1x1(16, 32)
        self.conv_head2 = resnet.conv1x1(32, 16)
        # third layer
        self.res3 = ResBlock(16, 32, stride=2, downsample=nn.Conv2d(16, 32, 1, stride=2), gate=self.gate)
        self.conv3 = resnet.conv1x1(32, 64)
        self.conv_head3 = resnet.conv1x1(64, 32)
        # pool
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

    def forward(self, x):
        """
        :param x: [B, C, H, W] C = 3, H, W % 64 == 0
        :return:
        score map        [B, 1, H, W]
        local desc map 0 [B, 3, H, W]
        local desc map 1 [B, 16, H/8, W/8]
        desc map         [B, 32, H/64, W/64]
        """
        # backbone feature
        layer1 = self.res1(x)
        layer2 = self.res2(layer1)  # 1/2
        layer2 = self.pool4(layer2)  # 1/4
        layer3 = self.res3(layer2)  # 1/2
        layer3 = self.pool4(layer3)  # 1/4
        # head
        x1 = self.gate(self.conv1(layer1))
        x1 = self.conv_head1(x1)
        scores_map = torch.sigmoid(x1[:, 3, :, :]).unsqueeze(1)
        x1 = torch.sigmoid(x1[:, :-1, :, :])
        x2 = self.gate(self.conv2(layer2))
        x2 = self.conv_head2(x2)
        x3 = self.gate(self.conv3(layer3))
        x3 = self.conv_head3(x3)
        # upsample and concat feature
        x3_up = F.interpolate(x3, scale_factor=8, mode='bilinear', align_corners=True)
        x2_up = torch.cat([x3_up, x2], dim=1)
        x2_up = F.interpolate(x2_up, scale_factor=8, mode='bilinear', align_corners=True)
        desc = torch.cat([x2_up, x1], dim=1)
        return scores_map, x1, x2, x3, desc


if __name__ == '__main__':
    model = ML_Point()
    x = torch.randn(1, 3, 512, 512)
    score, x1, x2, x3, desc = model(x)
    print(score.shape)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(desc.shape)

