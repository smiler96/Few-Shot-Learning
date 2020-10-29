import torch
import numpy as np
from importlib import import_module
from data.sampler import CategoriesSampler
from data.dataset import MiniImageNet
from torch.utils.data import DataLoader
import utils
import random
from loguru import logger

def test(args):
    # set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # check the args setting
    args = utils.check_test_args(args)

    test_dataset = MiniImageNet(phase='test', root=args.data_root)
    test_sampler = CategoriesSampler(labels=test_dataset.label, n_batch=args.batch,
                                      n_cls=args.n_way, n_per=args.n_shot + args.n_query)

    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler,
                              num_workers=args.n_threads, pin_memory=True)

    # 4. model setting
    module = import_module('model.' + args.model.lower())
    model = module.wrapper(**(vars(args)))
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['model_state_dict'])

    acc_all, acc_val = model.eval_loop(test_loader, 1, eval_name='Test')
    acc_std = np.std(np.array(acc_all))
    # logger.info('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
    logger.info(f'Test accuracy: {acc_val*100:4.2f}% +- {100 * 1.96 * acc_std / np.sqrt(args.batch):4.2f}%')

if __name__ == "__main__":
    from option import args
    args.batch = 600
    args.log_interval = 5
    test(args=args)