import torch
import torch.nn as nn
from model.meta_template import MetaTemplate
from importlib import import_module
from model.backbone.common import ConvBlock
from model.utils import ont_hot


def wrapper(**kwargs):
    return RelationNet(**kwargs)


class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(ConvNet, self).__init__()

        self.encoder = nn.Sequential(*[
            ConvBlock(inc=x_dim, outc=hid_dim, bn=True, act='relu', pooling='max'),
            ConvBlock(inc=hid_dim, outc=hid_dim, bn=True, act='relu', pooling='max'),
            ConvBlock(inc=hid_dim, outc=hid_dim, bn=True, act='relu', pooling=None),
            ConvBlock(inc=hid_dim, outc=z_dim, bn=True, act='relu', pooling=None),
        ])

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x


class RelationBlock(nn.Module):
    def __init__(self, fea_dim=64 * 2, loss_type='mse'):
        super(RelationBlock, self).__init__()

        self.compress = nn.Sequential(
            ConvBlock(inc=fea_dim, outc=64, k=3, s=1, p=1, bn=True, act='relu', pooling='max'),
            ConvBlock(inc=64, outc=64, k=3, s=1, p=1, bn=True, act='relu', pooling='max'),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = [
            nn.Linear(64, 8),
            nn.ReLU(True),
            nn.Linear(8, 1)
        ]
        if loss_type == 'mse':
            self.fc.append(nn.Sigmoid())
        elif loss_type == 'crossentropy':
            self.fc.append(nn.Softmax())
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        x = self.compress(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RelationNet(MetaTemplate):
    def __init__(self, **kwargs):
        super(RelationNet, self).__init__(**kwargs)

        if 'resnet' in kwargs['encoder'].lower():
            fea_dim = 512
        else:
            fea_dim = 64
        self.metrixer = RelationBlock(fea_dim * 2, kwargs['loss_type'])

        self.loss_type = kwargs['loss_type']
        if self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    def set_forward(self, X):
        p = self.n_shot * self.n_way
        # [(shot+query) * way, C, H, W]
        feature = self.encoder(X)
        _, C, H, W = feature.size()

        feat_Support = feature[:p]
        feat_Query = feature[p:]

        # model the proto for every class [way, C, H, W]
        protos = feat_Support.reshape(self.n_shot, self.n_way, C, H, W).mean(dim=0)

        # [way*query, way, C, H, W]
        protos = protos.unsqueeze(0).repeat(self.n_query * self.n_way, 1, 1, 1, 1)
        feat_Query = feat_Query.unsqueeze(0).repeat(self.n_way, 1, 1, 1, 1)
        feat_Query = torch.transpose(feat_Query, 0, 1)

        relation_pairs = torch.cat((protos, feat_Query), 2).view(-1, C * 2, H, W)

        # calculate the scores
        scores = self.metrixer(relation_pairs)
        scores = scores.view(-1, self.n_way)

        return scores

    def set_forward_loss(self, X):
        scores = self.set_forward(X)
        labels = self.prepare_label()
        if self.loss_type == 'mse':
            one_hot_labels = ont_hot(labels, self.n_way)
            loss = self.criterion(scores, one_hot_labels)
        else:
            loss = self.criterion(scores, labels)
        acc = self.calculate_acc(scores, labels)

        return loss, acc

    def get_encoder_module(self, name='convnet'):
        name = name.lower()
        if name == 'convnet':
            encoder = ConvNet(x_dim=3, hid_dim=64, z_dim=64)
            layers = list(encoder.children())
        elif 'resnet' in name:
            module = import_module('model.backbone.resnet')
            encoder = module.wrapper(n=int(name.split('resnet')[1]))
            layers = list(encoder.children())[:-1]
        else:
            raise NotImplementedError
        encoder = Encoder(layers)
        return encoder


if __name__ == "__main__":
    from option import args

    args.use_gpu = False
    args.encoder = 'resnet10'
    model = RelationNet(**vars(args))
    import numpy as np

    x = np.random.uniform(0, 1, [(args.n_shot + args.n_query) * args.n_way, 3, 84, 84]).astype(np.float32)
    x = torch.tensor(x)

    # loss = nn.L1Loss()
    # Adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999))
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        scores = model.set_forward(x)
        loss, acc = model.set_forward_loss(x)
    print(f'scores.shape: {scores.shape}')
    print(f'acc: {acc}')
