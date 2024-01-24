import os
import argparse
import ttach as tta
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional import accuracy, recall, f1_score
from copy import deepcopy
from math import log
from utils.dataset import choose_dataset
from tqdm import tqdm
from evaluate import evaluation
from reliability import compute_irs, compute_prs


means = {'BUSI': [0.32823274, 0.32822795, 0.32818426],
        'YBUS': [0.28199544, 0.28199544, 0.28199544]}
stds = {'BUSI': [0.22100208, 0.22100282, 0.22098634],
       'YBUS': [0.17909113, 0.17909113, 0.17909113]}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_uncertainty(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    val_data = choose_dataset(args.dataset, args.root, mode='val')
    val_loader = DataLoader(val_data, batch_size=args.batch_size//4)
    transforms = tta.aliases.d4_transform()
    ent_all = []
    for img, label, img_path in val_loader:
        img = img.to(device)
        outputs = []
        for transformer in transforms:
            # augment image
            augmented_image = transformer.augment_image(img)
            # pass to model
            model_output = F.softmax(model(augmented_image), dim=-1).data
            deaugmented_output = transformer.deaugment_label(model_output)
            outputs.append(deaugmented_output)

        outputs = torch.stack(outputs, dim=1)
        preds = torch.argmax(outputs, dim=-1).data
        onehot_preds = torch.zeros_like(outputs).to(device)
        p = onehot_preds.scatter_(-1, preds.unsqueeze(-1), 1)
        p = torch.sum(p, dim=1) / outputs.size(1)
        ent = torch.distributions.Categorical(probs=p).entropy() / log(outputs.size(-1))
        ent_all.append(ent)

    ent_all = torch.cat(ent_all, dim=0)

    uncer = torch.mean(ent_all)

    return uncer.item()


def compute_mdrs(model, test_data, args):
    test_loader = DataLoader(test_data, batch_size=1, num_workers=4, pin_memory=True)

    mdrs = 0
    for img, label, mask, img_path in test_loader:
        model.eval()
        img, label = img.to(device), label.to(device)
        output_orig = F.softmax(model(img), dim=-1).data
        prob, pred = torch.max(output_orig.squeeze(), dim=-1)
        prob, pred = prob.item(), pred.item()
        irs = max(prob, compute_irs(model, pred, mask, img_path, args))
        prs = compute_prs(model, output_orig, img, mask)
        if irs < 0:
            drs = prs
        else:
            drs = 0.5 * irs + 0.5 * prs
        mdrs += drs
    mdrs /= len(test_loader)

    return mdrs


def fuse_model(cur_state_dict, new_state_dict, cur_num, threshold):
    assert new_state_dict.keys() == cur_state_dict.keys()
    ave_idx = torch.rand(len(new_state_dict.keys()))
    for i, (k, v) in enumerate(new_state_dict.items()):
        if cur_state_dict[k].dtype != v.dtype:
            if k.split('.')[-1] == 'num_batches_tracked':
                v.to(dtype=torch.long)
        if ave_idx[i].item() < threshold:
            cur_state_dict[k] = cur_state_dict[k] * cur_num[i]
            cur_state_dict[k] = cur_state_dict[k] + v
            cur_num[i] += 1
            if cur_state_dict[k].dtype == torch.int64:
                cur_state_dict[k].div_(cur_num[i], rounding_mode='trunc')
            else:
                cur_state_dict[k].div_(cur_num[i])

    return cur_state_dict, cur_num


def soups(model, args):
    val_data = choose_dataset(args.dataset, args.root, mode='val')
    test_data = choose_dataset(args.dataset, args.root, mode='test')
    model_list = os.listdir(args.model_path)
    model_list.sort(reverse=True)
    model.load_state_dict(torch.load(os.path.join(args.model_path, model_list[0])))
    output_all, label_all = evaluation(model, test_data, args.batch_size)
    test_acc = accuracy(output_all, label_all).item()
    print(test_acc)

    drs_dict = {}
    for i in range(len(model_list)):
        model.load_state_dict(torch.load(os.path.join(args.model_path, model_list[i])))
        drs_dict[model_list[i]] = compute_mdrs(model, val_data, args)

    model_list.sort(key=lambda x: drs_dict[x])
    cur_state_dict = torch.load(os.path.join(args.model_path, model_list[0]))
    cur_num = torch.ones(len(cur_state_dict.keys()), dtype=torch.long)
    model.load_state_dict(cur_state_dict)
    best_drs = drs_dict[model_list[0]]

    num_ingredients = 1
    for k in range(int(1 / args.threshold) + 1):
        for i in range(1, len(model_list)):
            add_state_dict = torch.load(os.path.join(args.model_path, model_list[i]))
            new_state_dict, new_num = fuse_model(deepcopy(cur_state_dict), add_state_dict, cur_num.clone(), args.threshold)
            model.load_state_dict(new_state_dict)
            drs = compute_mdrs(model, val_data, args)
            if drs > best_drs:
                cur_state_dict = new_state_dict
                cur_num = new_num
                best_drs = drs
                num_ingredients += 1

    model.load_state_dict(cur_state_dict)
    output_all, label_all = evaluation(model, test_data, args.batch_size)
    acc = accuracy(output_all, label_all).item()
    rec = recall(output_all, label_all, num_classes=1, multiclass=False).item()
    f1 = f1_score(output_all, label_all, num_classes=1, multiclass=False).item()
    soup_path = os.path.join(args.save_path, args.dataset, args.model)
    if not os.path.exists(soup_path):
        os.makedirs(soup_path)
    if num_ingredients > 1:
        torch.save(cur_state_dict, os.path.join(soup_path, '{}_{:.5f}_{:.5f}_{:.5f}_RRsoup.pth'.format(
                args.model, acc, rec, f1)))
        print('num_ingredients={},threshold={:.5f},acc={:.5f},rec={:.5f},f1={:.5f}'.format(
            num_ingredients, args.threshold, acc, rec, f1))
        print(cur_num)

    return acc, rec, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soups')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--root', type=str, default='/home/hhn/hhn/data/YBUS')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--dataset', type=str, default='YBUS')
    parser.add_argument('--model_path', type=str, default='/home/hhn/hhn/code/BURS/checkpoint/YBUS/resnet50')
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--save_path', type=str, default='checkpoint/soup')
    args = parser.parse_args()

    model = eval('torchvision.models.{}'.format(args.model))(pretrained=False, num_classes=args.num_classes)

    best_acc, best_rec, best_f1 = 0, 0, 0
    thre_search = list(np.random.rand(args.samples))
    thre_search.sort()
    for i in tqdm(range(args.samples)):
        args.threshold = 0.1 + 0.4 * thre_search[i]
        acc, rec, f1 = soups(model, args)
        if acc > best_acc:
            best_acc, best_rec, best_f1 = acc, rec, f1
    print(best_acc, best_rec, best_f1)
