from math import ceil

import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        kernel_size=3,
        stride=1,
        padding=0,
        groups=1,
        bn=True,
        act=True,
        bias=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.batch_norm = nn.BatchNorm2d(n_out) if bn else nn.Identity()
        self.activation = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, n_in, reduced_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_in, reduced_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, n_in, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.seq(x)


class StochasticDepth(nn.Module):
    def __init__(self, survival_prob=0.8):
        super().__init__()
        self.p = survival_prob

    def forward(self, x):
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p
        return torch.div(x, self.p) * binary_tensor


class MBConvN(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        kernel_size=3,
        stride=1,
        expansion_factor=6,
        reduction=4,
        survival_prob=0.8,
    ):
        super().__init__()
        self.skip_connection = stride == 1 and n_in == n_out
        intermediate_channels = int(n_in * expansion_factor)
        padding = (kernel_size - 1) // 2
        reduced_dim = int(n_in // reduction)

        self.expand = (
            nn.Identity()
            if expansion_factor == 1
            else CNNBlock(n_in, intermediate_channels, kernel_size=1)
        )
        self.depthwise_conv = CNNBlock(
            intermediate_channels,
            intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=intermediate_channels,
        )
        self.se = SqueezeExcitation(intermediate_channels, reduced_dim=reduced_dim)
        self.pointwise_conv = CNNBlock(
            intermediate_channels, n_out, kernel_size=1, act=False
        )
        self.drop_layers = StochasticDepth(survival_prob=survival_prob)

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.pointwise_conv(x)
        if self.skip_connection:
            x = self.drop_layers(x)
            x += residual
        return x


class EfficientNet(nn.Module):
    # (width_mult, depth_mult, resolution, dropout_rate)
    CONFIGS = {
        "b0": (1.0, 1.0, 224, 0.2),
        "b1": (1.0, 1.1, 240, 0.2),
    }

    def __init__(self, version="b0", num_classes=200):
        super().__init__()
        width_mult, depth_mult, _, dropout_rate = self.CONFIGS[version]
        last_channel = ceil(1280 * width_mult)
        self.features = self._feature_extractor(width_mult, depth_mult, last_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x.view(x.shape[0], -1))
        return x

    def _feature_extractor(self, width_mult, depth_mult, last_channel):
        channels = 4 * ceil(int(32 * width_mult) / 4)
        layers = [CNNBlock(3, channels, kernel_size=3, stride=2, padding=1)]
        in_channels = channels

        kernels = [3, 3, 5, 3, 5, 5, 3]
        expansions = [1, 6, 6, 6, 6, 6, 6]
        num_channels = [16, 24, 40, 80, 112, 192, 320]
        num_layers = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]

        scaled_num_channels = [4 * ceil(int(c * width_mult) / 4) for c in num_channels]
        scaled_num_layers = [int(d * depth_mult) for d in num_layers]

        for i in range(len(scaled_num_channels)):
            layers += [
                MBConvN(
                    in_channels if repeat == 0 else scaled_num_channels[i],
                    scaled_num_channels[i],
                    kernel_size=kernels[i],
                    stride=strides[i] if repeat == 0 else 1,
                    expansion_factor=expansions[i],
                )
                for repeat in range(scaled_num_layers[i])
            ]
            in_channels = scaled_num_channels[i]

        layers.append(
            CNNBlock(in_channels, last_channel, kernel_size=1, stride=1, padding=0)
        )
        return nn.Sequential(*layers)
