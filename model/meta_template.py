import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from importlib import import_module


class MetaTemplate(nn.Module):
    def __init__(self, **kwargs):
        '''
        abstract class of meta learning
        :param encoder: backbone model for feature encoding, covnet / resnet | str type
        :param n_way: class number per meta task
        :param n_shot: support sample number per class for every meta task
        :param n_query: query sample number per class for every meta task
        :param log: log information in local
        '''
        super(MetaTemplate, self).__init__()

        assert kwargs['encoder'] is not None
        self.encoder = self.get_encoder_module(name=kwargs['encoder'])

        self.n_way = kwargs['n_way']
        self.n_shot = kwargs['n_shot']
        self.n_query = kwargs['n_query']
        self.log = kwargs['need_log']
        self.log_interval = kwargs['log_interval']

        if ('tblog' in kwargs) and (kwargs['tblog'] is not None):
            self.writer = SummaryWriter(log_dir=kwargs['tblog'])
        else:
            self.writer = None

        if kwargs['use_gpu'] and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.deterministic = True
            logger.info("CUDA visible devices: " + str(torch.cuda.device_count()))
            logger.info("CUDA Device Name: " + str(torch.cuda.get_device_name(self.device)))
        else:
            self.device = torch.device('cpu')

        self.criterion = None
        self.optimizer = None
        if ('optimizer' in kwargs) and (kwargs['optimizer'] is not None):
            self.optimizer = self.get_optimizer(kwargs['optimizer'], **kwargs)
        self.lr_scheduler = None
        if (self.optimizer is not None) and ('lr_scheduler' in kwargs) and (kwargs['lr_scheduler'] is not None):
            self.lr_scheduler = self.get_lr_scheduler(kwargs['lr_scheduler'], **kwargs)

    def forward(self, x):
        feature = self.encoder(x)
        return feature

    @abstractmethod
    def set_forward(self, X):
        '''
        propagate input data X forward
        :param X: [shot+query, C, H, W]
        :return: scores of query samples with respective of shot(support) samples, [query * way, way]
        '''
        pass

    @abstractmethod
    def set_forward_loss(self, X):
        '''
        must be implemented in subclass for calculate loss
        :param x:
        :return: loss, acc | tensor(require_grad=True), float
        '''
        pass

    def train_loop(self, train_loader, epoch):
        '''
        loop one epoch for training
        :param train_loader: auxiliary dataset
        :param epoch: current epoch
        :param optimizer: model optimizer
        :return: average loss \ acc
        '''
        self.to(self.device).train()
        with torch.autograd.detect_anomaly():
            for i, batch in enumerate(train_loader):
                X = batch[0]
                X = X.to(self.device, non_blocking=True)

                loss, acc = self.set_forward_loss(X)
                # gradient decent
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('Train/Loss', loss.item(), global_step=epoch * len(train_loader) + i)
                self.writer.add_scalar('Train/Accuracy', acc, global_step=epoch * len(train_loader) + i)

                if self.log and ((i + 1) % self.log_interval == 0):
                    logger.info(
                        f'Epoch-{epoch}-Batch-{i + 1}/{len(train_loader)}, train loss: {loss.item():.4f}, acc: {acc:.4f}')
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch=epoch)
                self.writer.add_scalar('Train/LR', self.lr_scheduler.get_lr()[0], epoch)

    def eval_loop(self, eval_loader, epoch, eval_name='Test'):
        '''
        test every episode
        :param test_loader: test dataset
        :param epoch: current epoch
        :return: accs
        '''
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            accs = []
            for i, batch in enumerate(eval_loader):
                X = batch[0]
                X = X.to(self.device, non_blocking=True)

                scores = self.set_forward(X)
                labels = self.prepare_label()
                acc = self.calculate_acc(scores, labels)

                accs.append(acc)
                if self.log and ((i + 1) % self.log_interval == 0):
                    logger.info(f'Epoch-{epoch}-Batch-{i + 1}/{len(eval_loader)}, {eval_name.lower()} acc: {acc:.4f}')

            avg_acc = np.mean(np.array(accs))
            if self.writer is not None:
                self.writer.add_scalar(f'{eval_name}/Accuracy', avg_acc, global_step=epoch)

            if self.log:
                logger.info(f'Epoch-{epoch}, {eval_name} acc: {avg_acc:.4f}')

        return accs, avg_acc

    def prepare_label(self):
        labels = torch.arange(self.n_way).repeat(self.n_query).type(torch.LongTensor)
        labels = labels.to(self.device)
        return labels

    def calculate_acc(self, scores, labels):
        '''
        calculate the acc of the query samples based on the support samples
        :param scores: query samples' scores with the way clusters, [query * way, way]
        :return: acc, float number
        '''
        preds = torch.argmax(scores, dim=1)
        acc = torch.mean((labels == preds).type(torch.FloatTensor).to(self.device)).item()
        return acc

    def get_encoder_module(self, name='ConvNet'):
        name = name.lower()
        if name == 'convnet':
            module = import_module('model.backbone.convnet')
            encoder = module.wrapper(in_c=3, h_dim=64, z_dim=64)
        elif 'resnet' in name:
            module = import_module('model.backbone.resnet')
            encoder = module.wrapper(n=int(name.split('resnet')[1]))
        else:
            raise NotImplementedError
        return encoder

    def get_optimizer(self, name, **kwargs):
        if name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=kwargs['lr'], weight_decay=kwargs['weight_decay'],
                                         betas=kwargs['betas'], eps=kwargs['epsilon'])
            logger.info('Use Adam Optimizer.')
        elif name.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=kwargs['lr'], weight_decay=kwargs['weight_decay'],
                                        nesterov=True,
                                        momentum=kwargs['momentum'])
            logger.info('Use SGD Optimizer.')
        else:
            raise ValueError(f"{name} optimizer has not been implemented yet.")
        return optimizer

    def get_lr_scheduler(self, name, **kwargs):
        assert self.optimizer is not None
        if name is None:
            return None
        decay_step = [int(s) for s in kwargs['decay_step'].split('-')]
        if name.lower() == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=decay_step[0],
                gamma=kwargs['gamma']
            )
            logger.info('Use Step Scheduler.')
        elif name.lower() == 'multistep':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=decay_step,
                gamma=kwargs['gamma'],
            )
            logger.info('Use MultiStep Scheduler.')
        elif name.lower() == 'cosineannealing':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                kwargs['epochs'],
                eta_min=0  # a tuning parameter
            )
            logger.info('Use CosineAnnealing Scheduler.')
        else:
            raise ValueError(f"{name} learning rate scheduler has not been implemented yet.")

        return lr_scheduler


