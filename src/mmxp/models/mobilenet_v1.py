import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=9, input_channels=3, width_mult=1.0):
        super().__init__()

        def c(channels):  # round and enforce min 1 channel
            return max(1, int(channels * width_mult))

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, c(32), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(c(32), c(64), stride=1),
            DepthwiseSeparableConv(c(64), c(128), stride=2),
            DepthwiseSeparableConv(c(128), c(128), stride=1),
            DepthwiseSeparableConv(c(128), c(256), stride=2),
            DepthwiseSeparableConv(c(256), c(256), stride=1),
            DepthwiseSeparableConv(c(256), c(512), stride=2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c(512), num_classes)
        )

    def forward(self, x):
        return self.model(x)
