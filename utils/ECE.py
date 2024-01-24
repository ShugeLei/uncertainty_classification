import numpy as np
import torch
from utils.compute_ece import CalibrationMetric
from torch.nn.functional import one_hot


def ECE_bin(pred, gt, num_bins=10, bin_method='EM'):
    """
    expected calibration error with given number of bins
    :param pred: [N, H, W, C] or [N, C]
    :param gt: [N, H, W] or [N]
    :param bin_method: 'EM' means equal-mass binning. 'EW' means equal-width binning.
    :return: expected calibration error
    """
    N = len(pred)

    if len(gt.shape) > 1:
        pred = torch.flatten(pred, start_dim=1, end_dim=2)
        gt = torch.flatten(gt, start_dim=1, end_dim=2)
        conf, pred_label = torch.max(pred, dim=-1)
        res = torch.eq(pred_label, gt).long()
        conf = torch.mean(conf, dim=-1)
        res = torch.mean(res.float(), dim=-1)
    else:
        conf, pred_label = torch.max(pred, dim=-1)
        res = torch.eq(pred_label, gt).long()
    conf, indices = torch.sort(conf, dim=0)

    if bin_method == 'EM':
        bins = torch.zeros((num_bins, 2))
        s = N // num_bins
        flag = N % num_bins
        bin_acc, bin_conf, k, b, ece = 0, 0, 0, 0, 0
        k_bins = []
        for i in range(N):
            if flag > 0:
                bin_size = s + 1
            else:
                bin_size = s
            bin_acc += res[indices[i]].item()
            bin_conf += conf[i].item()
            k += 1
            if k == bin_size or i == N - 1:
                bins[b][0], bins[b][1] = bin_acc / k, bin_conf / k
                ece += abs(bins[b][0] - bins[b][1]) * k
                k_bins.append(k)
                b += 1
                flag -= 1
                bin_acc, bin_conf, k = 0, 0, 0

    if bin_method == 'EW':
        bounds = np.linspace(0, 1, num_bins+1)
        bins = torch.zeros((num_bins, 2))
        bin_acc, bin_conf, k, ece = 0, 0, 0, 0
        b = min(int(conf[0].item() * num_bins), num_bins - 1)
        k_bins = [[] for _ in range(num_bins)]
        kk = torch.zeros(num_bins)
        for i in range(N):
            if bounds[b] <= conf[i] < bounds[b+1]:
                bins[b][0] += res[indices[i]].item()
                bins[b][1] += conf[i].item()
                kk[b] += 1
                k_bins[b].append(conf[i].item())
            elif conf[i] == bounds[-1]:
                bins[-1][0] += res[indices[i]].item()
                bins[-1][1] += conf[i].item()
                kk[-1] += 1
                k_bins[b].append(conf[i].item())
            else:
                b += 1
        for i in range(num_bins):
            if bins[i][0] > 0:
                bins[i][0] /= kk[i]
            if bins[i][1] > 0:
                bins[i][1] /= kk[i]
        ece = torch.sum(torch.mul(torch.abs(bins[:, 0] - bins[:, 1]), kk))
    ece /= N

    return ece, bins


def ece(pred, gt, num_bins=10, ce_method='em_ece_sweep'):
    """
    Compute ECE.
    :param pred: [N, H, W, C] or [N, C]
    :param gt: [N, H, W] or [N]
    :param num_bins: number of bins
    :param ce_method: 'em_ece_bin', 'ew_ece_bin', 'em_ece_sweep' or 'ew_ece_sweep'
    :return: ECE
    """
    if len(gt.shape) > 1:
        pred1 = torch.flatten(pred, start_dim=0, end_dim=2).cpu().numpy()
        gt1 = torch.flatten(gt, start_dim=0, end_dim=2).cpu().numpy()
    else:
        pred1 = pred.cpu().numpy()
        gt1 = one_hot(gt, num_classes=pred.shape[1]).cpu().numpy()
    if 'sweep' in ce_method:
        compute_ce = CalibrationMetric(ce_type=ce_method)
    else:
        compute_ce = CalibrationMetric(ce_type=ce_method, num_bins=num_bins)
    ece, bin_fx_y = compute_ce.compute_error(pred1, gt1)

    return ece, bin_fx_y


if __name__ == '__main__':
    inputs = torch.rand((16, 3))
    labels = torch.randint(low=0, high=3, size=(16,))
    o1, o2 = ECE_bin(inputs, labels, bin_method='EW')
