import torch.nn as nn

acts = {
    'relu': nn.ReLU(True),
    'leaky_relu': nn.LeakyReLU(negative_slope=0.2),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid()
}

poolings = {
    'max': lambda k: nn.MaxPool2d(k),
    'mean': lambda k: nn.AvgPool2d(k),
}

class ConvBlock(nn.Module):
    def __init__(self, inc, outc, k=3, s=1, p=1, bn=False, act=None, pooling=None):
        super(ConvBlock, self).__init__()
        self.op = []
        self.op.append(nn.Conv2d(inc, outc, k, s, p))
        if bn:
            self.op.append(nn.BatchNorm2d(outc))
        if act is not None:
            self.op.append(acts[act])
        if pooling is not None:
            self.op.append(poolings[pooling](2))

        self.op = nn.Sequential(*self.op)

    def forward(self, x):
        x = self.op(x)
        return x
