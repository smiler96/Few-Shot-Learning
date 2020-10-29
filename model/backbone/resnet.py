import torch.nn as nn


def conv1x1(inc, outc, stride=1):
    return nn.Conv2d(inc, outc, kernel_size=1, stride=stride, bias=False)


def conv3x3(inc, outc, stride=1):
    return nn.Conv2d(inc, outc, kernel_size=3, stride=stride, padding=1, bias=False)


class BaseBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, plances, stride=1, downsample=None):
        super(BaseBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, plances, stride)
        self.bn1 = nn.BatchNorm2d(plances)
        self.relu = nn.ReLU(True)

        self.conv2 = conv3x3(plances, plances)
        self.bn2 = nn.BatchNorm2d(plances)
        self.downsample = downsample

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)

        res = self.conv2(res)
        res = self.bn2(res)
        if self.downsample is not None:
            x = self.downsample(x)

        x = self.relu(x + res)
        return x


class BottleBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, plances, stride=1, downsample=None):
        super(BottleBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, plances)
        self.bn1 = nn.BatchNorm2d(plances)

        self.conv2 = conv3x3(plances, plances, stride)
        self.bn2 = nn.BatchNorm2d(plances)

        self.conv3 = conv1x1(plances, plances * self.expansion)
        self.bn3 = nn.BatchNorm2d(plances * self.expansion)

        self.relu = nn.ReLU(True)

        self.downsample = downsample

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.relu(self.bn2(self.conv2(res)))
        res = self.bn3(self.conv3(res))

        if self.downsample is not None:
            x = self.downsample(x)

        x = self.relu(x + res)

        return x



class ResNet(nn.Module):
    def __init__(self, block=BaseBlock, layers=[2, 2, 2, 2]):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(*[
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, n_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        return x


def resnet10(**kwargs):
    return ResNet(BaseBlock, [1, 1, 1, 1], **kwargs)


def resnet18(**kwargs):
    return ResNet(BaseBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BaseBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(BottleBlock, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(BottleBlock, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(BottleBlock, [3, 8, 36, 3], **kwargs)

def wrapper(**args):
    if args['n'] == 10:
        return resnet10()
    elif args['n'] == 18:
        return resnet18()
    elif args['n'] == 34:
        return resnet34()
    elif args['n'] == 50:
        return resnet50()
    elif args['n'] == 101:
        return resnet101()
    elif args['n'] == 152:
        return resnet152()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    import numpy as np
    import torch
    import torchsummary

    model = resnet50()
    x = np.random.uniform(0, 1, [1, 3, 224, 224]).astype(np.float32)
    x = torch.tensor(x)

    y = model(x)
    print(y.shape)

