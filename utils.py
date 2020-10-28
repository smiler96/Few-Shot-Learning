import os
import shutil
import torch
from loguru import logger

def check_path(path, reset=False):
    if reset and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def check_saves(args):
    # check the root of saving items
    check_path(args.save_root, False)

    name = f'{args.model}_{args.encoder}_way-{args.n_way}_shot-{args.n_shot}_query-{args.n_query}-{args.dataset}'
    # check the tblog
    args.tblog = os.path.join(args.save_root, 'tblog', name)
    check_path(args.tblog, True)

    # check the check-point
    check_path(os.path.join(args.save_root, 'ckpt'), False)
    args.ckpt = os.path.join(args.save_root, 'ckpt', name)

    # logger file
    args.logger = os.path.join(args.save_root, 'ckpt', f'{name}.txt')
    if os.path.exists(args.logger):
        os.remove(args.logger)
    logger.add(args.logger, rotation="200 MB", backtrace=True, diagnose=True)
    logger.info(str(args))
    return args


def check_optimizer(args, model):
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                     betas=args.betas, eps=args.epsilon)
        logger.info('Use Adam Optimizer.')
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    nesterov=True,
                                    momentum=args.momentum)
        logger.info('Use SGD Optimizer.')

    decay_step = [int(s) for s in args.decay_step.split('-')]
    if args.lr_scheduler == 'Step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=decay_step[0],
            gamma=args.gamma
        )
        logger.info('Use Step Scheduler.')
    elif args.lr_scheduler == 'MultiStep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=decay_step,
            gamma=args.gamma,
        )
        logger.info('Use MultiStep Scheduler.')
    elif args.lr_scheduler == 'CosineAnnealing':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.epochs,
            eta_min=0  # a tuning parameter
        )
        logger.info('Use CosineAnnealing Scheduler.')
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler