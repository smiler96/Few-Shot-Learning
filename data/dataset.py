import os
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MiniImageNet(Dataset):
    def __init__(self, phase="train", root=None, transform=None):
        '''
        miniImageNet dataset
        :param phase: choice = [train, val, test]
        :param root: data root path
        '''
        assert root is not None

        self.input = []
        self.label = []
        lb = -1
        self.idx_to_class = []
        with open(os.path.join(root, f'{phase}.csv'), 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            name, cls = line.strip().split(',')

            self.input.append(os.path.join(root, 'images/' + name))
            if cls not in self.idx_to_class:
                self.idx_to_class.append(cls)
                lb += 1
            self.label.append(lb)

        self.len = len(self.input)
        logger.info(f'Load {self.len} {phase} samples from {os.path.join(root, "images/")}.')

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(84),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = self.input[idx]
        label = self.label[idx]
        image = self.transform(Image.open(img_path).convert('RGB'))
        return image, label


class TieredImageNet(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

if __name__ == "__main__":
    mini = MiniImageNet('train', root='D:/Dataset/miniImageNet/')
    from torch.utils.data import DataLoader
    from data.sampler import CategoriesSampler
    mini_loader = DataLoader(mini,
                             batch_sampler=CategoriesSampler(mini.label_lists, n_batch=100, n_cls=5, n_per=5))
    for i, batch in enumerate(mini_loader):
        print(batch.shape)