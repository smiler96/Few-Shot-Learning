import torch.nn as nn
from model.backbone.common import ConvBlock

def wrapper(**args):
    return ConvNet(args['in_c'], args['h_dim'], args['z_dim'])

class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(ConvNet, self).__init__()

        self.encoder = nn.Sequential(*[
            ConvBlock(inc=x_dim, outc=hid_dim, bn=True, act='relu', pooling='max'),
            ConvBlock(inc=hid_dim, outc=hid_dim, bn=True, act='relu', pooling='max'),
            ConvBlock(inc=hid_dim, outc=hid_dim, bn=True, act='relu', pooling='max'),
            ConvBlock(inc=hid_dim, outc=z_dim, bn=True, act='relu', pooling='max'),
        ])
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x

if __name__ == "__main__":
    import numpy as np
    import torch
    import torchsummary

    model = ConvNet()
    print(torchsummary.summary(model, (3, 84, 84), device='cpu'))

    x = np.random.uniform(0, 1, [2, 3, 84, 84]).astype(np.float32)
    x = torch.tensor(x)

    # loss = nn.L1Loss()
    # Adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999))
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y = model(x)
    print(prof)
    print(y.shape)
