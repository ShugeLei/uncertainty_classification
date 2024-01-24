import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import os
from PIL import Image
import cv2


class BUS(Dataset):
    def __init__(self, root, task='classification', mode='train', mask_root=None):
        assert task in ['classification', 'reliability']
        self.task = task
        self.img_root = os.path.join(root, 'images')
        self.mask_root = mask_root
        if 'BUSI' in root:
            list_root = os.path.join(root, 'list', task)
            mean = [0.32823274, 0.32822795, 0.32818426]
            std = [0.22100208, 0.22100282, 0.22098634]
        elif 'YBUS' in root:
            list_root = os.path.join(root, 'list')
            mean = [0.28199544, 0.28199544, 0.28199544]
            std = [0.17909113, 0.17909113, 0.17909113]
        with open(os.path.join(list_root, mode + '.txt'), 'r') as f:
            self.data_list = [i.split(',') for i in f.read().splitlines()]
        if mode == 'train':
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)])
        else:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)])

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.data_list[idx][0])
        label = self.data_list[idx][1]
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        if 'classification' in self.task:
            return img, int(label), img_path
        else:
            mask = Image.open(os.path.join(self.mask_root, self.data_list[idx][0].replace('.jpg', '_pred.png'))).convert('L')
            mask = T.Compose([T.Resize((224, 224)), T.ToTensor()])(mask)
            return img, int(label), mask, img_path

    def __len__(self):
        return len(self.data_list)


def choose_dataset(dataset_name, data_path, task='classification', mode='test', mask_root=None):
    if 'BUS' in dataset_name:
        data = BUS(data_path, task=task, mode=mode, mask_root=os.path.join(mask_root, dataset_name))
        return data
