from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU, Mish, ELU, RReLU, GELU
from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.parent.as_posix())


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        use_bn=True,
        activation="ReLU",
        inplace=True,
    ):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups,
                bias=False,
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation in ["ReLU", "Mish", "ELU", "RReLU"]:
            layers.append(getattr(nn, activation)(inplace=inplace))
        elif activation == "GELU":
            layers.append(getattr(nn, activation)())
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class ResidualBlock(nn.Module):
    def __init__(self, module):
        super(ResidualBlock, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class AdaptiveMaxPool2d(nn.Module):
    def __init__(self, output_size=(1, 1)):
        super(AdaptiveMaxPool2d, self).__init__()
        self.layer = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):
        return self.layer(x)


class MaxDropout(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(MaxDropout, self).__init__()
        if not (0 <= dropout_prob <= 1):
            raise ValueError(
                f"Dropout probability must be between 0 and 1, but got {dropout_prob}"
            )
        self.dropout_prob = 1 - dropout_prob

    def forward(self, x):
        if not self.training:
            return x
        x_normalized = (x - x.min()) / (x.max() - x.min())
        mask = x_normalized > self.dropout_prob
        return x.masked_fill(mask, 0)


class ResNet9(nn.Module):
    def __init__(self, num_classes=10, use_bn=True, initial_channels=64, inplace=True):
        super(ResNet9, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                use_bn=use_bn,
                inplace=inplace,
            ),
            ConvBlock(
                64,
                128,
                kernel_size=5,
                stride=2,
                padding=2,
                use_bn=use_bn,
                inplace=inplace,
            ),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(128, 128, use_bn=use_bn, inplace=inplace),
                    ConvBlock(128, 128, use_bn=use_bn, inplace=inplace),
                )
            ),
            ConvBlock(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                use_bn=use_bn,
                inplace=inplace,
            ),
            nn.MaxPool2d(2),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(256, 256, use_bn=use_bn, inplace=inplace),
                    ConvBlock(256, 256, use_bn=use_bn, inplace=inplace),
                )
            ),
            ConvBlock(
                256,
                128,
                kernel_size=3,
                stride=1,
                padding=0,
                use_bn=use_bn,
                inplace=inplace,
            ),
            AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )

    def forward(self, x):
        return self.layers(x)


class ResNet9WithDropout(nn.Module):
    def __init__(
        self,
        num_classes=10,
        use_bn=True,
        initial_channels=64,
        dropout_prob=0.3,
        inplace=True,
    ):
        super(ResNet9WithDropout, self).__init__()
        self.dropout = MaxDropout(dropout_prob)
        self.layers = nn.Sequential(
            ConvBlock(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                use_bn=use_bn,
                inplace=inplace,
            ),
            ConvBlock(
                64,
                128,
                kernel_size=5,
                stride=2,
                padding=2,
                use_bn=use_bn,
                inplace=inplace,
            ),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(128, 128, use_bn=use_bn, inplace=inplace),
                    ConvBlock(128, 128, use_bn=use_bn, inplace=inplace),
                )
            ),
            self.dropout,
            ConvBlock(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                use_bn=use_bn,
                inplace=inplace,
            ),
            nn.MaxPool2d(2),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(256, 256, use_bn=use_bn, inplace=inplace),
                    ConvBlock(256, 256, use_bn=use_bn, inplace=inplace),
                )
            ),
            self.dropout,
            ConvBlock(
                256,
                128,
                kernel_size=3,
                stride=1,
                padding=0,
                use_bn=use_bn,
                inplace=inplace,
            ),
            AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )

    def forward(self, x):
        return self.layers(x)


class ResNet9_MaxDropout(nn.Module):
    def __init__(self, num_classes=10, use_bn=True, channel=64):
        super(ResNet9_MaxDropout, self).__init__()
        self.dropout = MaxDropout()  # dropout训练
        self.layers = nn.Sequential(
            ConvBlock(3, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 128, kernel_size=5, stride=2, padding=2),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(128, 128),
                    ConvBlock(128, 128),
                )
            ),
            self.dropout,
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(256, 256),
                    ConvBlock(256, 256),
                )
            ),
            self.dropout,
            ConvBlock(256, 128, kernel_size=3, stride=1, padding=0),
            AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )

    def forward(self, x):
        return self.layers(x)


class simp_ResNet9(nn.Module):
    def __init__(self, num_classes=10, use_bn=True, channel=64, activation="ReLU"):
        super(simp_ResNet9, self).__init__()
        self.activation = activation
        self.layers = nn.Sequential(
            ConvBlock(
                3, 64, kernel_size=3, stride=1, padding=1, activation=activation
            ),
            ConvBlock(
                64, 128, kernel_size=5, stride=2, padding=2, activation=activation
            ),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(
                        128, 32, kernel_size=1, padding=0, activation=activation
                    ),
                    ConvBlock(32, 32, activation=activation),
                    ConvBlock(
                        32, 128, kernel_size=1, padding=0, activation=activation
                    ),
                )
            ),
            ConvBlock(
                128, 256, kernel_size=3, stride=1, padding=1, activation=activation
            ),
            nn.MaxPool2d(2),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(
                        256, 64, kernel_size=1, padding=0, activation=activation
                    ),
                    ConvBlock(64, 64, activation=activation),
                    ConvBlock(
                        64, 256, kernel_size=1, padding=0, activation=activation
                    ),
                )
            ),
            ConvBlock(
                256, 128, kernel_size=3, stride=1, padding=0, activation=activation
            ),
            AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )

    def forward(self, x):
        return self.layers(x)


class simp_ResNet9_MaxDropout(nn.Module):
    def __init__(self, num_classes=10, use_bn=True, channel=64, activation="ReLU"):
        super(simp_ResNet9_MaxDropout, self).__init__()
        self.dropout = MaxDropout()  # dropout训练
        self.layers = nn.Sequential(
            ConvBlock(
                3, 64, kernel_size=3, stride=1, padding=1, activation=activation
            ),
            ConvBlock(
                64, 128, kernel_size=5, stride=2, padding=2, activation=activation
            ),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(
                        128, 32, kernel_size=1, padding=0, activation=activation
                    ),
                    ConvBlock(32, 32, activation=activation),
                    ConvBlock(
                        32, 128, kernel_size=1, padding=0, activation=activation
                    ),
                )
            ),
            self.dropout,
            ConvBlock(
                128, 256, kernel_size=3, stride=1, padding=1, activation=activation
            ),
            nn.MaxPool2d(2),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(
                        256, 64, kernel_size=1, padding=0, activation=activation
                    ),
                    ConvBlock(64, 64, activation=activation),
                    ConvBlock(
                        64, 256, kernel_size=1, padding=0, activation=activation
                    ),
                )
            ),
            self.dropout,
            ConvBlock(
                256, 128, kernel_size=3, stride=1, padding=0, activation=activation
            ),
            AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )

    def forward(self, x):
        return self.layers(x)


class AMP2d(torch.nn.Module):
    def __init__(self, H=1, W=1):
        super(AMP2d, self).__init__()
        self.layer = nn.AdaptiveMaxPool2d((H, W))

    def forward(self, x):
        return self.layer(x)


class simp_ResNet9_k7(nn.Module):
    def __init__(self, num_classes=10, if_bn=True, channel=64, activation="ReLU"):
        super(simp_ResNet9_k7, self).__init__()
        self.activation = activation
        self.layer = nn.Sequential(
            ConvBlock(
                3, 128, kernel_size=7, stride=2, padding=3, activation=activation
            ),
            # torch.nn.MaxPool2d(2),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(128, 32, kernel_size=1, padding=0, activation=activation),
                    ConvBlock(32, 32, activation=activation),
                    ConvBlock(32, 128, kernel_size=1, padding=0, activation=activation),
                )
            ),
            ConvBlock(
                128, 256, kernel_size=3, stride=1, padding=1, activation=activation
            ),
            nn.MaxPool2d(2),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(256, 64, kernel_size=1, padding=0, activation=activation),
                    ConvBlock(64, 64, activation=activation),
                    ConvBlock(64, 256, kernel_size=1, padding=0, activation=activation),
                )
            ),
            ConvBlock(
                256, 128, kernel_size=3, stride=1, padding=0, activation=activation
            ),
            AMP2d(1, 1),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class simp_ResNet9_k333(nn.Module):
    def __init__(self, num_classes=10, if_bn=True, channel=64, activation="ReLU"):
        super(simp_ResNet9_k333, self).__init__()
        self.activation = activation
        self.layer = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1, activation=activation),
            ConvBlock(
                32, 64, kernel_size=3, stride=1, padding=2, activation=activation
            ),
            ConvBlock(
                64, 128, kernel_size=3, stride=2, padding=2, activation=activation
            ),
            # torch.nn.MaxPool2d(2),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(128, 32, kernel_size=1, padding=0, activation=activation),
                    ConvBlock(32, 32, activation=activation),
                    ConvBlock(32, 128, kernel_size=1, padding=0, activation=activation),
                )
            ),
            ConvBlock(
                128, 256, kernel_size=3, stride=1, padding=1, activation=activation
            ),
            nn.MaxPool2d(2),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(256, 64, kernel_size=1, padding=0, activation=activation),
                    ConvBlock(64, 64, activation=activation),
                    ConvBlock(64, 256, kernel_size=1, padding=0, activation=activation),
                )
            ),
            ConvBlock(
                256, 128, kernel_size=3, stride=1, padding=0, activation=activation
            ),
            AMP2d(1, 1),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class simp_ResNet9_k33(nn.Module):
    def __init__(self, num_classes=10, if_bn=True, channel=64, activation="ReLU"):
        super(simp_ResNet9_k33, self).__init__()
        self.activation = activation
        self.layer = nn.Sequential(
            ConvBlock(3, 64, kernel_size=3, stride=1, padding=1, activation=activation),
            ConvBlock(
                64, 128, kernel_size=3, stride=2, padding=2, activation=activation
            ),
            # torch.nn.MaxPool2d(2),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(128, 32, kernel_size=1, padding=0, activation=activation),
                    ConvBlock(32, 32, activation=activation),
                    ConvBlock(32, 128, kernel_size=1, padding=0, activation=activation),
                )
            ),
            ConvBlock(
                128, 256, kernel_size=3, stride=1, padding=1, activation=activation
            ),
            nn.MaxPool2d(2),
            ResidualBlock(
                nn.Sequential(
                    ConvBlock(256, 64, kernel_size=1, padding=0, activation=activation),
                    ConvBlock(64, 64, activation=activation),
                    ConvBlock(64, 256, kernel_size=1, padding=0, activation=activation),
                )
            ),
            ConvBlock(
                256, 128, kernel_size=3, stride=1, padding=0, activation=activation
            ),
            AMP2d(1, 1),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, batch_norm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        ) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.act(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, if_bn=True, channel=64):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel) if if_bn else nn.Identity(),
            nn.ReLU()
        )
        self.layer1 = self._make_layer(channel, channel, 1, if_bn)
        self.layer2 = self._make_layer(channel, channel * 2, 2, if_bn)
        self.layer3 = self._make_layer(channel * 2, channel * 4, 2, if_bn)
        self.layer4 = self._make_layer(channel * 4, channel * 8, 2, if_bn)
        self.fc = nn.Linear(channel * 8, num_classes)

    def _make_layer(self, in_channels, out_channels, stride, if_bn):
        return nn.Sequential(
            ResBlock(in_channels, out_channels, stride, if_bn),
            ResBlock(out_channels, out_channels, 1, if_bn)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def init_weights_(m):
    """
    Initializes weights of m according to Xavier normal method.

    :param m: module
    :return:
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class VGG_A(nn.Module):
    """VGG_A model

    size of Linear layers is smaller since input assumed to be 32x32x3, instead of
    224x224x3
    """

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()

        self.features = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes))

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


class VGG_A_Light(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        '''
        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        '''
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        # x = self.stage3(x)
        # x = self.stage4(x)
        # x = self.stage5(x)
        x = self.classifier(x.view(-1, 32 * 8 * 8))
        return x


class VGG_A_Dropout(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

class VGG_A_BatchNorm(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()

        self.features = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(512 * 1 * 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes))

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)
