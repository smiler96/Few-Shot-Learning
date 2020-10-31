import argparse
import os

parser = argparse.ArgumentParser(description='FSL')

# Train or Test
parser.add_argument('--phase', type=str, default='train',
                    help='chose the phase for the model, train or test',
                    choices=['train', 'test'])
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from last train status')
parser.add_argument('--seed', type=int, default=666,
                    help='random seed')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=2,
                    help='number of threads for data loading')
parser.add_argument('--use_gpu', action='store_false',
                    help='use GPU')

# Data specifications
parser.add_argument('--dataset', type=str, default='MiniImageNet',
                    help='dataset for experiment')
parser.add_argument('--in_c', type=int, default=3,
                    help='dataset channels')
parser.add_argument('--data_root', type=str, default='D:/Dataset/miniImageNet/',
                    help='dataset root')

parser.add_argument('--augment', action='store_true',
                    help='use data augmentation, random horizontal flips and 90 rotations')

# Model specifications
parser.add_argument('--model', default='protonet',
                    help='model name')
parser.add_argument('--encoder', default='convnet',
                    help="model's base encoder")
# RealtionNet
parser.add_argument('--fc_dim', type=int, default=1600,
                    help="1600 for convnet encoder and 512 for resnet10-34 with image size=84")

# Training specifications
parser.add_argument('--n_way', type=int, default=5,
                    help='the number of classes during training')
parser.add_argument('--n_shot', type=int, default=5,
                    help='the number of samples per class during training')
parser.add_argument('--n_query', type=int, default=15,
                    help='the number of samples per class to evaluate the performance during training')


# Optimization specifications
parser.add_argument('--epochs', type=int, default=600,
                    help='number of epochs to train')
parser.add_argument('--batch', type=int, default=100,
                    help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--lr_scheduler', type=str, default=None,
                    help='learning rate scheduler to use (Step | MultiStep | Cosine)')
parser.add_argument('--decay_step', type=str, default='60',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')

parser.add_argument('--optimizer', default='Adam',
                    choices=('SGD', 'Adam', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')

parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')

parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Test specification
parser.add_argument('--eval_way', type=int, default=5,
                    help='the number of classes during testing')
parser.add_argument('--eval_shot', type=int, default=5,
                    help='the number of samples per class during testing')
parser.add_argument('--eval_query', type=int, default=15,
                    help='the number of samples per class to evaluate the performance during testing')

# Loss specifications
# RealtionNet
parser.add_argument('--loss_type', default='mse',
                    help='loss type for relation net')

# Log specifications
parser.add_argument('--save_root', type=str, default='saves',
                    help='the root of the save, including the tblogs/ weights/ loggers/ train_visulization')
parser.add_argument('--need_log', action='store_false',
                    help='log is or not need')
parser.add_argument('--log_interval', default=40,
                    help='log acc / loss every log_interval batch')

args = parser.parse_args()
args.save_root = os.path.abspath(args.save_root)

if __name__ == "__main__":
    print(vars(args))