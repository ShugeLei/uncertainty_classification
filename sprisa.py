import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import cv2
from copy import deepcopy
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm


means = {'BUSI': [0.32823274, 0.32822795, 0.32818426],
        'YBUS': [0.28199544, 0.28199544, 0.28199544]}
stds = {'BUSI': [0.22100208, 0.22100282, 0.22098634],
       'YBUS': [0.17909113, 0.17909113, 0.17909113]}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BUS(Dataset):
    def __init__(self, root, mode='test'):
        assert mode in ['train', 'val', 'test']
        self.img_root = os.path.join(root, 'images')
        list_root = os.path.join(root, 'list')
        with open(os.path.join(list_root, mode + '.txt'), 'r') as f:
            self.data_list = [i.split(',') for i in f.read().splitlines()]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.data_list[idx][0])
        label = self.data_list[idx][1]
        img = cv2.resize(cv2.imread(img_path, 1), (256, 256))

        return img, int(label), img_path

    def __len__(self):
        return len(self.data_list)


class Masked(Dataset):
    def __init__(self, img, masks, mean, std):
        self.img = img
        self.masks = masks
        self.img_trans = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=mean, std=std)])
        self.mask_trans = T.Compose([
            T.ToTensor(),
            T.Resize((32, 32)),
            T.Resize((224, 224))])

    def __getitem__(self, idx):
        mask = self.masks[idx]
        img = self.img_trans(self.img)
        mask = self.mask_trans(mask)
        inp = img*mask
        return inp, mask.squeeze()

    def __len__(self):
        return len(self.masks)


def rise(model, img, pred, mean, std, n_mask=5000, batch_size=512):
    masks = []
    for _ in range(n_mask):
        mask = np.random.uniform(0, 1, size=(16, 16)) < 0.5
        masks.append((mask*255).astype(np.uint8))
    model = model.to(device)
    model.eval()
    input_data = Masked(img, masks, mean=mean, std=std)
    masked_loader = DataLoader(input_data, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for i, (masked_img, mask) in enumerate(masked_loader):
            masked_img, mask = masked_img.to(device), mask.to(device)
            outputs = F.softmax(model(masked_img), dim=1).data
            scores = outputs[:, pred].unsqueeze(-1).unsqueeze(-1)
            if i == 0:
                mask_tensors = torch.sum(mask * scores, dim=0)
                n_pixels = torch.sum(mask, dim=0)
            else:
                mask_tensors += torch.sum(mask * scores, dim=0)
                n_pixels += torch.sum(mask, dim=0)
        attribution = mask_tensors / n_pixels
        a, b = torch.max(attribution), torch.min(attribution)
        attribution -= b
        attribution /= a - b
    return attribution


def generate_masks(img, n_mask=4000):
    slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=30, ruler=20)
    slic.iterate(10)
    masks = slic.getLabels()
    palette = list(set(list(masks.flatten())))
    n = len(palette)
    mask_list = []
    for i in palette:
        m = np.where(masks == i, 0, 255).astype(np.uint8)
        mask_list.append(m)
    n_sample = round((n_mask - n) / int(n/2))
    ss = np.random.uniform(0, 1, size=(int(n/2), n_sample, n))
    for i in range(int(n/2)):
        for j in range(n_sample):
            hit = np.asarray(ss[i, j] < (n-2-i)/n).nonzero()[0].tolist()
            if len(hit) < 5:
                break
            m = np.ones_like(masks, dtype=np.uint8) * 255
            hit_erase = list(set(palette) - set(hit))
            for k in hit_erase:
                m[np.where(masks == k)] = 0
            mask_list.append(m)
            # mask_list = np.concatenate([mask_list, np.expand_dims(m, axis=0)], axis=0)
    mask_list = np.stack(mask_list, axis=0)
    print(len(mask_list))
    return mask_list


def sp_risa(model, image, pred, mean, std, batch_size=200):
    masks = generate_masks(image, 3000)
    input_data = Masked(image, masks, mean=mean, std=std)
    mask_loader = DataLoader(input_data, batch_size=batch_size, shuffle=False)
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    for i, (masked_img, mask) in enumerate(mask_loader):
        masked_img, mask = masked_img.to(device), mask.to(device)
        outputs = F.softmax(model(masked_img), dim=1).data
        scores = 1 - outputs[:, pred]
        scores = scores.unsqueeze(-1).unsqueeze(-1)
        mask = torch.ones((1, 1, 1)).to(device) - mask
        if i == 0:
            mask_tensors = torch.sum(mask * scores, dim=0)
            n_pixels = torch.sum(mask, dim=0)
        else:
            mask_tensors += torch.sum(mask * scores, dim=0)
            n_pixels += torch.sum(mask, dim=0)
    attribution = mask_tensors / n_pixels
    a, b = torch.max(attribution), torch.min(attribution)
    attribution -= b
    attribution /= a - b
    return attribution


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SP-RISA')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--root', type=str, default='/home/hhn/hhn/data/YBUS')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='YBUS')
    parser.add_argument('--model_path', type=str,
                        default='checkpoint/YBUS/resnet50/resnet50_0.90108_0.91642_0.91779_53_0.00008_16_1_pretrained.pth')
    parser.add_argument('--save_path', type=str, default='result/SP-RISA')
    args = parser.parse_args()

    model = eval('torchvision.models.{}'.format(args.model))(pretrained=False, num_classes=args.num_classes)
    test_data = BUS(args.root, mode='test')
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    mean, std = means[args.dataset], stds[args.dataset]

    for image, label, img_path in tqdm(test_data):
        img_tensor = T.Compose([T.ToTensor(), T.Resize((224, 224)), T.Normalize(mean=mean, std=std)])(image)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_output = F.softmax(model(img_tensor), dim=-1).data
        pred = torch.argmax(img_output[0]).item()
        if pred == label:
            attribution = sp_risa(model, image, pred, mean, std, batch_size=args.batch_size)
            # attr_map = (attribution * 255).cpu().numpy().astype(np.uint8)
            # cv2.imwrite(os.path.join(args.save_path, 'attri', os.path.basename(img_path).replace('.jpg', '_sprisa.png')), attr_map)
            # attr_map = cv2.applyColorMap(attr_map, cv2.COLORMAP_JET)
            # s = attr_map.shape
            # img = cv2.resize(image, (s[0], s[1]))
            # attr_img = cv2.addWeighted(img, 0.7, attr_map, 0.3, 0)
            rise_map = rise(model, image, pred, mean, std) * 255
            # rise_map = cv2.applyColorMap(rise_map.cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
            # rise_img = cv2.addWeighted(img, 0.7, rise_map, 0.3, 0)
            # res1 = np.concatenate([img, rise_map, rise_img], axis=-2)
            # res2 = np.concatenate([img, attr_map, attr_img], axis=-2)
            # res = np.concatenate([res1, res2], axis=-3)
            # cv2.imwrite(os.path.join(args.save_path, 'figure', os.path.basename(img_path).replace('jpg', 'png')), res)
