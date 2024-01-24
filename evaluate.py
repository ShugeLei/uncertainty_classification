import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.dataset import choose_dataset
from utils.ECE import ece
from torchmetrics.functional import accuracy, recall, f1_score
from tqdm import tqdm


def evaluation(model, test_data, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    with torch.no_grad():
        output_all = []
        label_all = []

        for images, labels, img_paths in test_loader:
            model.eval()
            images, labels = images.to(device), labels.to(device)
            outputs = F.softmax(model(images), dim=-1).data
            output_all.append(outputs)
            label_all.append(labels)

        output_all = torch.cat(output_all, dim=0)
        label_all = torch.cat(label_all, dim=0)

    return output_all, label_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soups')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--root', type=str, default='/home/hhn/hhn/data/YBUS')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='YBUS')
    parser.add_argument('--model_path', type=str, default='checkpoint/YBUS/resnet50')
    parser.add_argument('--save_path', type=str, default='log/soups.txt')
    args = parser.parse_args()

    model = eval('torchvision.models.{}'.format(args.model))(pretrained=False, num_classes=args.num_classes)

    test_data = choose_dataset(args.dataset, args.root, task='classification', mode='test')
    model_list = os.listdir(args.model_path)
    model_list.sort(reverse=True)
    for weight_name in tqdm(model_list):
        weight_path = os.path.join(args.model_path, weight_name)
        model.load_state_dict(torch.load(weight_path))
        output_all, label_all = evaluation(model, test_data, args.batch_size)
        acc = accuracy(output_all, label_all).item()
        rec = recall(output_all, label_all, num_classes=1, multiclass=False).item()
        f1 = f1_score(output_all, label_all, num_classes=1, multiclass=False).item()
        with open('log/model_metrics.txt', 'a+') as f:
            f.write('{},{},{},{}\n'.format(weight_name, acc, rec, f1))
    # ece_bin_em, bin_em_data = ece(output_all, label_all, num_bins=10, ce_method='em_ece_bin')
    # ece_bin_ew, bin_ew_data = ece(output_all, label_all, num_bins=10, ce_method='ew_ece_bin')
    # ece_sweep_em, sweep_em_data = ece(output_all, label_all, ce_method='em_ece_sweep')
    # ece_sweep_ew, sweep_ew_data = ece(output_all, label_all, ce_method='ew_ece_sweep')
    # with open(args.save_path, 'a+') as f:
    #     f.write('{} {}:{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(
    #         args.dataset, args.model_path, acc, auc, f1, ece_bin_em, ece_bin_ew, ece_sweep_em, ece_sweep_ew))
    # with open('log/{}/reliability_diagram.txt'.format(args.dataset), 'a+') as f:
    #     f.write('{}\n'.format(args.model_path))
    #     f.write('bin_em\n{}\n{}\nbin_ew\n{}\n{}\nsweep_em\n{}\n{}\nsweep_ew\n{}\n{}\n'.format(
    #         ' '.join(str(i) for i in bin_em_data[0]), ' '.join(str(i) for i in bin_em_data[1]),
    #         ' '.join(str(i) for i in bin_ew_data[0]), ' '.join(str(i) for i in bin_ew_data[1]),
    #         ' '.join(str(i) for i in sweep_em_data[0]), ' '.join(str(i) for i in sweep_em_data[1]),
    #         ' '.join(str(i) for i in sweep_ew_data[0]), ' '.join(str(i) for i in sweep_ew_data[1])))
