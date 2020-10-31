import torch
import torch.nn as nn
from model.meta_template import MetaTemplate
from importlib import import_module
from model.backbone.common import ConvBlock

class MatchingNet(MetaTemplate):
    def __init__(self):
        super(MatchingNet, self).__init__()

    def set_forward(self, X):
        pass

    def set_forward_loss(self, X):
        pass