import os
import argparse
import random

import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.functional.classification import accuracy, recall
from utils.dataset import choose_dataset
from torchvision import transforms as T
from tqdm import tqdm
from utils.ECE import ece
from sprisa import sp_risa
import cv2
from pynvml import *
import matplotlib.pyplot as plt


plt.switch_backend('agg')


means = {'BUSI': [0.32823274, 0.32822795, 0.32818426],
        'YBUS': [0.28199544, 0.28199544, 0.28199544]}
stds = {'BUSI': [0.22100208, 0.22100282, 0.22098634],
       'YBUS': [0.17909113, 0.17909113, 0.17909113]}


def compute_prs(model, output_orig, img, mask):
    transform1 = T.Compose([T.RandomRotation((-30, 30)), T.Resize((224, 224))])
    transform2 = T.RandomCrop((224, 224))
    transform3 = T.Compose([T.RandomHorizontalFlip(1), T.Resize((224, 224))])
    transform4 = T.Compose([T.RandomRotation((-30, 30)), T.RandomCrop((224, 224))])
    transform5 = T.Compose([T.RandomRotation((-30, 30)), T.RandomHorizontalFlip(1), T.Resize((224, 224))])
    transform6 = T.Compose([T.RandomHorizontalFlip(1), T.RandomCrop((224, 224))])
    transform7 = T.Compose([T.RandomRotation((-30, 30)), T.RandomHorizontalFlip(1), T.RandomCrop((224, 224))])
    transforms = [transform1, transform2, transform3, transform4, transform5, transform6, transform7]
    mask_size = torch.sum(mask).item()
    outputs = [output_orig]
    for transformer in transforms:
        i = 0
        while i < 10:
            # augment image
            augmented_image = transformer(img)
            augmented_mask = transformer(mask)
            if torch.sum(augmented_mask.squeeze()).item() > mask_size-10:
                # pass to model
                model_output = F.softmax(model(augmented_image), dim=-1).data
                outputs.append(model_output)
                break
            i += 1

    outputs = torch.stack(outputs, dim=1)
    preds = torch.argmax(outputs, dim=-1).data
    onehot_preds = torch.zeros_like(outputs).to(img.device)
    p = onehot_preds.scatter_(-1, preds.unsqueeze(-1), 1)
    p = torch.sum(p, dim=1) / outputs.size(1)
    p = torch.clamp(p, 1e-5, 1)
    # prs = 1 - torch.distributions.Categorical(probs=p).entropy() / log(outputs.size(-1))
    prs = 1 + torch.sum(p * torch.log(p)) / torch.log(torch.Tensor([outputs.size(-1)]).to(img.device))

    return prs


def compute_irs(model, pred, mask, img_path, args):
    """
    img, mask: [C, H, W]
    """
    h, w = mask.size(-2), mask.size(-1)
    scaled_mask = T.Resize((round(1.21 * h), round(1.21 * w)))(mask)
    scaled_mask = T.CenterCrop((h, w))(scaled_mask)
    mask_behind = scaled_mask[0].cpu().numpy()
    contours, hierarchy = cv2.findContours(mask_behind, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lty, rby = h, 0
    for contour in contours:
        lty = min(lty, np.min(contour[:, :, 1]))
        rby = max(rby, np.max(contour[:, :, 1]))
    mask_h = rby - lty
    for _ in range(mask_h):
        mask_behind += np.concatenate([np.zeros((1, w), dtype=np.uint8), mask_behind[:-1]], axis=0)
    mask_pro = scaled_mask.squeeze() + torch.from_numpy(mask_behind)
    # mask_pro = scaled_mask.squeeze()
    # x = torch.max(torch.sum(mask, dim=0)).item()
    # if x == 0:
    #     return torch.Tensor([-1])
    # y2 = h - torch.sum(mask.squeeze(), dim=-1).nonzero()[-1].item()
    # for _ in range(min(int(x), y2)):
    #     mask_pro += torch.cat([torch.zeros([1, w]).to(mask.device), mask_pro[:-1]])
    mask_pro = torch.clamp(mask_pro, 0, 1)
    m_size = round(torch.sum(mask_pro).item())
    image_np = cv2.imread(img_path[0])
    attribution = sp_risa(model, image_np, pred, mean=means[args.dataset], std=stds[args.dataset], batch_size=args.batch_size)
    attribution_flatten = torch.flatten(attribution)
    _, indices = torch.topk(attribution_flatten, m_size)
    s = torch.zeros_like(attribution_flatten).to(mask.device)
    s[indices] = 1
    irs = torch.sum(s * torch.flatten(mask_pro)) / m_size

    return irs


def drs_tester(weight_name, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = eval('torchvision.models.{}'.format(args.model))(pretrained=False, num_classes=args.num_classes)
    model.load_state_dict(torch.load(os.path.join(args.model_path, weight_name)))
    model = model.to(device)
    model.eval()

    # test_data = choose_dataset(args.dataset, args.root, task='reliability', mode='test', mask_root=args.mask_root)
    test_data = choose_dataset(args.dataset, args.root, mode='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    output_all = []
    label_all = []
    prs_all = []
    drs_all = []
    screen_all = []
    screen_labels = []
    mdrs = 0

    for img, label, mask, img_path in test_loader:
        img, label = img.to(device), label.to(device)
        oo = model(img)
        output_orig = F.softmax(oo, dim=-1).data
        prob, pred = torch.max(output_orig.squeeze(), dim=-1)
        prob, pred = prob.item(), pred.item()
        conf_output = F.softmax(oo/args.temperature, dim=-1).data
        output_all.append(conf_output)
        label_all.append(label)
        irs = compute_irs(model, pred, mask, img_path, args)
        irs = max(irs, prob)
        prs = compute_prs(model, output_orig, img, mask)
        drs = 0.5*irs + 0.5*prs

        mdrs += drs
        prs_output, drs_output = torch.zeros_like(output_orig), torch.zeros_like(output_orig)
        prs_output[:, pred], prs_output[:, 1 - pred] = prs, 0
        drs_output[:, pred], drs_output[:, 1 - pred] = drs, 0
        prs_all.append(prs_output.to(device))
        drs_all.append(drs_output.to(device))

    output_all = torch.cat(output_all, dim=0)
    label_all = torch.cat(label_all, dim=0)
    prs_all = torch.cat(prs_all, dim=0)
    drs_all = torch.cat(drs_all, dim=0)

    print('mDRS={}'.format(mdrs/len(test_loader)))
    print('confidence')
    conf_bin_ew, bin_ew_data = ece(output_all, label_all, num_bins=10, ce_method='ew_ece_bin')
    conf_bin_em, bin_em_data = ece(output_all, label_all, num_bins=10, ce_method='em_ece_bin')
    conf_sweep_ew, sweep_ew_data = ece(output_all, label_all, ce_method='ew_ece_sweep')
    conf_sweep_em, sweep_em_data = ece(output_all, label_all, ce_method='em_ece_sweep')
    print('ece_bw={},ece_bm={},ece_sw={},ece_sm={}'.format(conf_bin_ew, conf_bin_em, conf_sweep_ew, conf_sweep_em))

    print('1-uncertainty')
    uncer_bin_ew, bin_ew_data = ece(prs_all, label_all, num_bins=10, ce_method='ew_ece_bin')
    uncer_bin_em, bin_em_data = ece(prs_all, label_all, num_bins=10, ce_method='em_ece_bin')
    uncer_sweep_ew, sweep_ew_data = ece(prs_all, label_all, ce_method='ew_ece_sweep')
    uncer_sweep_em, sweep_em_data = ece(prs_all, label_all, ce_method='em_ece_sweep')
    print('ece_bw={},ece_bm={},ece_sw={},ece_sm={}'.format(uncer_bin_ew, uncer_bin_em, uncer_sweep_ew, uncer_sweep_em))

    print('DRS')
    drs_bin_ew, bin_ew_data = ece(drs_all, label_all, num_bins=10, ce_method='ew_ece_bin')
    drs_bin_em, bin_em_data = ece(drs_all, label_all, num_bins=10, ce_method='em_ece_bin')
    drs_sweep_ew, sweep_ew_data = ece(drs_all, label_all, ce_method='ew_ece_sweep')
    drs_sweep_em, sweep_em_data = ece(drs_all, label_all, ce_method='em_ece_sweep')
    print('ece_bw={},ece_bm={},ece_sw={},ece_sm={}'.format(drs_bin_ew, drs_bin_em, drs_sweep_ew, drs_sweep_em))
    acc = accuracy(output_all, label_all).item()
    rec = recall(output_all, label_all, num_classes=1, multiclass=False).item()
    print(acc, rec)

    for i in range(len(output_all)):
        if torch.max(drs_all[i]) > args.threshold:
            screen_all.append(output_all[i].unsqueeze(0))
            screen_labels.append(label_all[i].unsqueeze(0))
    screen_all = torch.cat(screen_all, dim=0)
    screen_labels = torch.cat(screen_labels, dim=0)
    print('screen, threshold={}'.format(args.threshold))
    acc_screen = accuracy(screen_all, screen_labels).item()
    rec_screen = recall(screen_all, screen_labels, num_classes=1, multiclass=False).item()
    print(acc_screen, rec_screen, len(screen_all))
    res = {'conf_ece_sm': conf_sweep_em,
           'uncer_ece_sm': uncer_sweep_em,
           'drs_ece_sm': drs_sweep_em}

    return res, acc_screen, rec_screen, len(screen_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soups')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--root', type=str, default='/home/hhn/hhn/data/YBUS')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='YBUS')
    parser.add_argument('--model_path', type=str, default='checkpoint/YBUS/resnet50')
    parser.add_argument('--mask_root', type=str, default='/media/yuexin/Unet/result')
    parser.add_argument('--temperature', type=float, default=8)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--num_GPU', type=int, default=3)
    args = parser.parse_args()

    # nvmlInit()
    # deviceCount = nvmlDeviceGetCount()
    # gpus = []
    # while True:
    #     for i in range(deviceCount):
    #         handle = nvmlDeviceGetHandleByIndex(i)
    #         free_memory = nvmlDeviceGetMemoryInfo(handle).free
    #         if free_memory / 1024 ** 2 > 9900:
    #             gpus.append(i)
    #     if len(gpus) >= args.num_GPU:
    #         nvmlShutdown()
    #         break
    # gpus.sort(reverse=True)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(_) for _ in gpus])

    weight_name = 'resnet50_0.90108_0.91642_0.91779_53_0.00008_16_1_pretrained.pth'
    res, acc_screen, rec_screen, n_screen = drs_tester(weight_name, args)
    # with open('result/drs.txt', 'a+') as f:
    #     f.write('{},{},{},{}\n'.format(weight_name, res['conf_ece_sm'], res['uncer_ece_sm'], res['drs_ece_sm']))
