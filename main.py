from option import args
from train import train
from test import test

if __name__ == "__main__":
    if args.phase == 'train':
        train(args)
    else:
        test(args)