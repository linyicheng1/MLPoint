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
    def __init__(self, params):
        super().__init__()
        c0 = params['c0']
        c1 = params['c1']
        h0 = params['h0']
        c2 = params['c2']
        h1 = params['h1']
        c3 = params['c3']
        c4 = params['c4']
        h2 = params['h2']
        self.gate = nn.ReLU(inplace=True)
        # first layer
        self.res1 = ResBlock(c0, c1, 1, downsample=nn.Conv2d(c0, c1, 1), gate=self.gate)
        self.conv1 = resnet.conv1x1(c1, h0)
        self.conv_head1 = resnet.conv1x1(h0, 4)
        # second layer
        self.res2 = ResBlock(c1, c2, stride=2, downsample=nn.Conv2d(c1, c2, 1, stride=2), gate=self.gate)
        self.conv2 = resnet.conv1x1(c2, h1)
        self.conv_head2 = resnet.conv1x1(h1, 32)
        # third layer
        self.res3 = ResBlock(c2, c3, stride=2, downsample=nn.Conv2d(c2, c3, 1, stride=2), gate=self.gate)
        self.res4 = ResBlock(c3, c4, stride=1, downsample=nn.Conv2d(c3, c4, 1, stride=1), gate=self.gate)
        self.conv3 = resnet.conv1x1(c4, h2)
        self.conv_head3 = resnet.conv1x1(h2, 64)
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
        layer4 = self.res4(layer3)  # 1
        # head 1
        x1 = self.gate(self.conv1(layer1))
        x1 = self.conv_head1(x1)
        scores_map = torch.sigmoid(x1[:, 3, :, :]).unsqueeze(1)
        desc_0 = torch.sigmoid(x1[:, :-1, :, :])
        # head 2
        x2 = self.gate(self.conv2(layer2))
        x2 = self.conv_head2(x2)
        desc_1 = F.normalize(x2, p=2, dim=1)
        # head 3
        x3 = self.gate(self.conv3(layer4))
        x3 = self.conv_head3(x3)
        desc_2 = F.normalize(x3, p=2, dim=1)

        return scores_map / torch.norm(scores_map) * 300, desc_0, desc_1, desc_2


if __name__ == '__main__':
    model = ML_Point()
    x = torch.randn(1, 3, 512, 512)
    score, x1, x2, x3, desc = model(x)
    print(score.shape)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(desc.shape)

