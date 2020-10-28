import torch
import numpy as np

class CategoriesSampler:
    def __init__(self, labels, n_batch, n_cls, n_per):
        '''
        random smaple n_batch n_cls*b_per samples for n_cls-way n_per-shot
        :param labels: list like [0, 1, 2, 3..], all classes is max(labels)
        :param n_batch: sample how many n_cls-way n_per-shot once
        :param n_cls: N-way
        :param n_per: K-shot
        '''

        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        labels = np.array(labels)
        self.m_ind = []
        for i in range(max(labels) + 1):
            ind = np.argwhere(labels == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            cls_ = torch.randperm(len(self.m_ind))[: self.n_cls]
            for c in cls_:
                c_indxs = self.m_ind[c]
                pos = torch.randperm(len(c_indxs))[: self.n_per]
                batch.append(c_indxs[pos])

            batch = torch.stack(batch)
            batch = batch.t()
            batch = batch.reshape(-1)
            yield batch

    # def __init__(self, labels, n_batch, n_cls, n_per):
    #     self.n_batch = n_batch
    #     self.n_cls = n_cls
    #     self.n_per = n_per
    #
    #     labels = np.array(labels)
    #     self.m_ind = []
    #     for i in range(max(labels) + 1):
    #         ind = np.argwhere(labels == i).reshape(-1)
    #         ind = torch.from_numpy(ind)
    #         self.m_ind.append(ind)
    #
    # def __len__(self):
    #     return self.n_batch
    #
    # def __iter__(self):
    #     for i_batch in range(self.n_batch):
    #         batch = []
    #         classes = torch.randperm(len(self.m_ind))[:self.n_cls]
    #         for c in classes:
    #             l = self.m_ind[c]
    #             pos = torch.randperm(len(l))[:self.n_per]
    #             batch.append(l[pos])
    #         batch = torch.stack(batch).t().reshape(-1)
    #         yield batch
