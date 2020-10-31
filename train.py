import torch
import numpy as np
from importlib import import_module
from data.sampler import CategoriesSampler
from data.dataset import MiniImageNet
from torch.utils.data import DataLoader
import utils
import random
from loguru import logger

def train(args):
    # set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # check the saves setting
    args = utils.check_saves(args)

    # 2. train / val dataset import
    train_dataset = MiniImageNet(phase='train', root=args.data_root)
    val_dataset = MiniImageNet(phase='val', root=args.data_root)

    # 3. batch sampler setting
    train_sampler = CategoriesSampler(labels=train_dataset.label, n_batch=args.batch,
                                      n_cls=args.n_way, n_per=args.n_shot + args.n_query)
    val_sampler = CategoriesSampler(labels=val_dataset.label, n_batch=args.batch * 4,
                                      n_cls=args.n_way, n_per=args.n_shot + args.n_query)

    # 3.1 data loader setting
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=args.n_threads, pin_memory=False)
    val_loader = DataLoader(train_dataset, batch_sampler=val_sampler,
                            num_workers=args.n_threads, pin_memory=False)

    # 4. model setting
    module = import_module('model.' + args.model.lower())
    model = module.wrapper(**(vars(args)))

    # 6.1 continue train or not
    if args.continue_train:
        ckpt = torch.load(args.ckpt)
        best_acc_val = ckpt['best_acc']
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model_state_dict'])
        model.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for state in model.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(model.device)

        logger.info(f'Successfully load check point from {args.ckpt}')

    else:
        best_acc_val = 0.0
        start_epoch = 0

    best_state_dict = model.state_dict()
    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()
        # 7. training batch loops
        model.train_loop(train_loader, epoch+1)

        # 7.1 validating batch loops
        _, acc_val = model.eval_loop(val_loader, epoch+1, eval_name='Val')

        # 9. save the best accuracy model weights
        model.cpu()
        if best_acc_val < acc_val:
            best_acc_val = acc_val
            best_state_dict = model.state_dict()
            logger.info(f'Current best model is at epoch-{epoch+1} and acc is: {best_acc_val:4.2f}')

        ckpt = {
            'epoch': epoch,
            # 'lr': model.lr_scheduler.get_lr()[0],
            'best_acc': best_acc_val,
            'model_state_dict': model.state_dict(),
            'best_state_dict': best_state_dict,
            'optimizer_state_dict': model.optimizer.state_dict()
        }
        torch.save(ckpt, args.ckpt)

if __name__ == "__main__":
    from option import args

    args.continue_train = False
    args.epochs = 600
    args.decay_step = '60'
    args.n_way = 5
    args.n_shot = 1
    args.n_query = 15
    train(args=args)
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     train(args=args)
    # print(prof)