import torch.nn as nn
from model.meta_template import MetaTemplate
from model.utils import euclidean_metrix

def wrapper(**kwargs):
    return ProtoNet(**kwargs)

class ProtoNet(MetaTemplate):
    def __init__(self, **kwargs):
        super(ProtoNet, self).__init__(**kwargs)

        self.criterion = nn.CrossEntropyLoss()

    def set_forward(self, X):
        p = self.n_shot * self.n_way
        # [(shot+query) * way, -1]
        feature = self.encoder(X)
        feat_Support = feature[:p, :]
        feat_Query = feature[p:, :]

        # model the proto for every class
        protos = feat_Support.reshape(self.n_shot, self.n_way, -1).mean(dim=0)

        # calculate the scores
        scores = euclidean_metrix(feat_Query, protos)

        return scores

    def set_forward_loss(self, X):
        scores = self.set_forward(X)
        # prepare query data labels [way * query]
        labels = self.prepare_label()

        # calculate the loss for backward
        loss = self.criterion(scores, labels)

        # measure the accuracy of the query data
        acc = self.calculate_acc(scores, labels)

        return loss, acc