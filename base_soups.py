import os
import shutil
import argparse
import torch
from evaluate import evaluation
import torchvision
from copy import deepcopy
from utils.dataset import choose_dataset
from utils.ECE import ece
from torchmetrics.functional import accuracy, auroc, f1_score


def fuse_model(cur_state_dict, new_state_dict, cur_num):
    assert new_state_dict.keys() == cur_state_dict.keys()
    for k, v in new_state_dict.items():
        if cur_state_dict[k].dtype != v.dtype:
            if k.split('.')[-1] == 'num_batches_tracked':
                v.to(dtype=torch.long)
        cur_state_dict[k] = cur_state_dict[k] * cur_num
        cur_state_dict[k] = cur_state_dict[k] + v
        if cur_state_dict[k].dtype == torch.int64:
            cur_state_dict[k].div_(cur_num + 1, rounding_mode='trunc')
        else:
            cur_state_dict[k].div_(cur_num + 1)

    return cur_state_dict, cur_num + 1


def uniform_soup(model, args):
    model_list = os.listdir(args.model_path)
    cur_state_dict = torch.load(os.path.join(args.model_path, model_list[0]))
    cur_num = 1
    for i in range(1, len(model_list)):
        add_state_dict = torch.load(os.path.join(args.model_path, model_list[i]))
        cur_state_dict, cur_num = fuse_model(cur_state_dict, add_state_dict, cur_num)
    model.load_state_dict(cur_state_dict)
    test_data = choose_dataset(args.dataset, args.root, mode='test')
    output_all, label_all = evaluation(model, test_data, args.batch_size)
    acc = accuracy(output_all, label_all).item()
    auc = auroc(output_all, label_all, average='macro', num_classes=args.num_classes).item()
    f1 = f1_score(output_all, label_all, average='macro', num_classes=args.num_classes).item()
    ece_bin_em, bin_em_data = ece(output_all, label_all, num_bins=10, ce_method='em_ece_bin')
    ece_bin_ew, bin_ew_data = ece(output_all, label_all, num_bins=10, ce_method='ew_ece_bin')
    ece_sweep_em, sweep_em_data = ece(output_all, label_all, ce_method='em_ece_sweep')
    ece_sweep_ew, sweep_ew_data = ece(output_all, label_all, ce_method='ew_ece_sweep')
    model_name = '{}_{:.5f}_{:.5f}_{:.5f}_uniform.pth'.format(
        args.model, acc, auc, f1)
    torch.save(cur_state_dict, os.path.join(args.model_path.replace(args.model, 'uniform_soup'), model_name))
    with open('log/{}/reliability_diagram.txt'.format(args.dataset), 'a+') as f:
        f.write('{},uniform_soup\n'.format(args.model))
        f.write('bin_em\n{}\n{}\nbin_ew\n{}\n{}\nsweep_em\n{}\n{}\nsweep_ew\n{}\n{}\n'.format(
            ' '.join(str(i) for i in bin_em_data[0]), ' '.join(str(i) for i in bin_em_data[1]),
            ' '.join(str(i) for i in bin_ew_data[0]), ' '.join(str(i) for i in bin_ew_data[1]),
            ' '.join(str(i) for i in sweep_em_data[0]), ' '.join(str(i) for i in sweep_em_data[1]),
            ' '.join(str(i) for i in sweep_ew_data[0]), ' '.join(str(i) for i in sweep_ew_data[1])))

    return acc, auc, f1, ece_bin_em, ece_bin_ew, ece_sweep_em, ece_sweep_ew


def greedy_soup(model, args):
    model_list = os.listdir(args.model_path)
    model_list.sort(reverse=True)
    cur_state_dict = torch.load(os.path.join(args.model_path, model_list[0]))
    cur_num = 1
    model.load_state_dict(cur_state_dict)
    val_data = choose_dataset(args.dataset, args.root, mode='val')
    output_all, label_all = evaluation(model, val_data, args.batch_size)
    best_acc = accuracy(output_all, label_all).item()
    for i in range(1, len(model_list)):
        add_state_dict = torch.load(os.path.join(args.model_path, model_list[i]))
        new_state_dict, new_num = fuse_model(deepcopy(cur_state_dict), add_state_dict, cur_num)
        model.load_state_dict(new_state_dict)
        output_all, label_all = evaluation(model, val_data, args.batch_size)
        acc = accuracy(output_all, label_all).item()
        if acc > best_acc:
            cur_state_dict = new_state_dict
            cur_num = new_num
            best_acc = acc
    model.load_state_dict(cur_state_dict)
    test_data = choose_dataset(args.dataset, args.root, mode='test')
    output_all, label_all = evaluation(model, test_data, args.batch_size)
    acc = accuracy(output_all, label_all).item()
    auc = auroc(output_all, label_all, average='macro', num_classes=args.num_classes).item()
    f1 = f1_score(output_all, label_all, average='macro', num_classes=args.num_classes).item()
    ece_bin_em, bin_em_data = ece(output_all, label_all, num_bins=10, ce_method='em_ece_bin')
    ece_bin_ew, bin_ew_data = ece(output_all, label_all, num_bins=10, ce_method='ew_ece_bin')
    ece_sweep_em, sweep_em_data = ece(output_all, label_all, ce_method='em_ece_sweep')
    ece_sweep_ew, sweep_ew_data = ece(output_all, label_all, ce_method='ew_ece_sweep')

    model_name = '{}_{:.5f}_{:.5f}_{:.5f}_greedy.pth'.format(
        args.model, acc, auc, f1)
    torch.save(cur_state_dict, os.path.join(args.model_path.replace(args.model, 'greedy_soup'), model_name))
    with open('log/{}/reliability_diagram.txt'.format(args.dataset), 'a+') as f:
        f.write('{},greedy_soup\n'.format(args.model))
        f.write('bin_em\n{}\n{}\nbin_ew\n{}\n{}\nsweep_em\n{}\n{}\nsweep_ew\n{}\n{}\n'.format(
            ' '.join(str(i) for i in bin_em_data[0]), ' '.join(str(i) for i in bin_em_data[1]),
            ' '.join(str(i) for i in bin_ew_data[0]), ' '.join(str(i) for i in bin_ew_data[1]),
            ' '.join(str(i) for i in sweep_em_data[0]), ' '.join(str(i) for i in sweep_em_data[1]),
            ' '.join(str(i) for i in sweep_ew_data[0]), ' '.join(str(i) for i in sweep_ew_data[1])))

    return acc, auc, f1, ece_bin_em, ece_bin_ew, ece_sweep_em, ece_sweep_ew


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baselines')
    parser.add_argument('--model', type=str, default='mobilenet_v2')
    parser.add_argument('--root', type=str, default='/home/hhn/hhn/data/ISIC2018_Task3_Training')
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='ISIC2018')
    parser.add_argument('--model_path', type=str, default='/home/hhn/hhn/code/X-ensemble/checkpoint/ISIC2018/mobilenet_v2')
    parser.add_argument('--save_path', type=str, default='log/baselines.txt')
    args = parser.parse_args()

    model = eval('torchvision.models.{}'.format(args.model))(pretrained=False, num_classes=args.num_classes)

    test_data = choose_dataset(args.dataset, args.root, mode='test')
    model_list = os.listdir(args.model_path)
    model_list.sort(reverse=True)
    # model_list.sort(reverse=True, key=lambda x: float(x.split('_')[args.split_idx]))
    best_val_model = model_list[0]
    print(best_val_model)
    model.load_state_dict(torch.load(os.path.join(args.model_path, best_val_model)))
    output_all, label_all = evaluation(model, test_data, args.batch_size)
    best_test_acc = accuracy(output_all, label_all).item()
    best_test_auc = auroc(output_all, label_all, average='macro', num_classes=args.num_classes).item()
    best_test_f1 = f1_score(output_all, label_all, average='macro', num_classes=args.num_classes).item()
    best_ece_bin_em, best_bin_em_data = ece(output_all, label_all, num_bins=10, ce_method='em_ece_bin')
    best_ece_bin_ew, best_bin_ew_data = ece(output_all, label_all, num_bins=10, ce_method='ew_ece_bin')
    best_ece_sweep_em, best_sweep_em_data = ece(output_all, label_all, ce_method='em_ece_sweep')
    best_ece_sweep_ew, best_sweep_ew_data = ece(output_all, label_all, ce_method='ew_ece_sweep')
    with open(args.save_path, 'a+') as f:
        f.write('{}\nbest individual on val set:{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(
            args.dataset, model_list[0], best_test_acc, best_test_auc, best_test_f1,
            best_ece_bin_em, best_ece_bin_ew, best_ece_sweep_em, best_ece_sweep_ew))
    with open('log/{}/reliability_diagram.txt'.format(args.dataset), 'a+') as f:
        f.write('{},best individual on val set\n'.format(best_val_model))
        f.write('bin_em\n{}\n{}\nbin_ew\n{}\n{}\nsweep_em\n{}\n{}\nsweep_ew\n{}\n{}\n'.format(
            ' '.join(str(i) for i in best_bin_em_data[0]), ' '.join(str(i) for i in best_bin_em_data[1]),
            ' '.join(str(i) for i in best_bin_ew_data[0]), ' '.join(str(i) for i in best_bin_ew_data[1]),
            ' '.join(str(i) for i in best_sweep_em_data[0]), ' '.join(str(i) for i in best_sweep_em_data[1]),
            ' '.join(str(i) for i in best_sweep_ew_data[0]), ' '.join(str(i) for i in best_sweep_ew_data[1])))

    uniform_acc, uniform_auc, uniform_f1, uniform_ece_bm, uniform_ece_bw, uniform_ece_sm, uniform_ece_sw = uniform_soup(model, args)
    greedy_acc, greedy_auc, greedy_f1, greedy_ece_bm, greedy_ece_bw, greedy_ece_sm, greedy_ece_sw = greedy_soup(model, args)
    print(len(model_list))
    with open(args.save_path, 'a+') as f:
        f.write('uniform soup:{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(
            uniform_acc, uniform_auc, uniform_f1, uniform_ece_bm, uniform_ece_bw, uniform_ece_sm, uniform_ece_sw))
        f.write('greedy soup:{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(
            greedy_acc, greedy_auc, greedy_f1, greedy_ece_bm, greedy_ece_bw, greedy_ece_sm, greedy_ece_sw))
