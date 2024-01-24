import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, dataset
from utils.dataset import choose_dataset
import torchvision
from torchmetrics.functional import accuracy, recall, f1_score, auroc
import matplotlib.pyplot as plt
from tqdm import tqdm


plt.switch_backend('agg')


def train(model, train_set, val_set, lr, batch_size, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    epochs = args.epochs

    if args.dataset == 'ISIC2018':
        loss_weight = np.array(args.loss_weight)
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(loss_weight)).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    best_acc = 0
    best_recall = 0
    best_f1 = 0
    best_epochs = epochs
    loss_train_all = []
    loss_val_all = []
    print('lr={},batch_size={},epochs={}'.format(lr, batch_size, epochs))
    for epoch in range(1, epochs + 1):
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        model.train()
        epoch_loss = 0
        for data, target, img_path in tqdm(train_loader):
            if torch.cuda.is_available():
                data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * target.size(0)

        epoch_loss /= len(train_set)
        loss_train_all.append(epoch_loss)

        with torch.no_grad():
            epoch_val_loss = 0
            output_all = []
            label_all = []

            for images, labels, img_paths in val_loader:
                model.eval()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss_val = criterion(outputs, labels)
                epoch_val_loss += loss_val.item() * labels.size(0)
                outputs = F.softmax(outputs, dim=1)
                output_all.append(outputs)
                label_all.append(labels)

        output_all = torch.cat(output_all, dim=0).to(device)
        label_all = torch.cat(label_all, dim=0).to(device)
        epoch_val_loss /= len(val_set)
        loss_val_all.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        if args.multi_GPU:
            model_weights = model.module.state_dict()
        else:
            model_weights = model.state_dict()

        acc = accuracy(output_all, label_all).item()
        rec = recall(output_all, label_all, num_classes=1, multiclass=False).item()
        f1 = f1_score(output_all, label_all, num_classes=1, multiclass=False).item()
        print('Epoch {}/{}:train_loss={:.5f} val_loss={:.5f} acc={:.5f} recall={:.5f} f1={:.5f}'.format(
            epoch, epochs, epoch_loss, epoch_val_loss, acc, rec, f1))
        if acc > args.acc and loss.item() < 0.1:
            model_name = '{}_{:.5f}_{:.5f}_{:.5f}_{}_{:.5f}_{}_{}'.format(
                args.model, acc, rec, f1, epoch, lr, batch_size, args.seed)
            if args.pretrain:
                model_name += '_pretrained'
            torch.save(model_weights, os.path.join(args.save_path, args.dataset, args.model, model_name+'.pth'))

        if acc > best_acc:
            best_acc = acc
            best_epochs = epoch + 10
        best_recall = max(rec, best_recall)
        best_f1 = max(f1, best_f1)

    if best_epochs > args.epochs:
        args.epochs = best_epochs

    plt.figure()
    plt.plot(loss_train_all)
    plt.plot(loss_val_all)
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.savefig('result/{}/{}_{:.5f}_{}_{}.png'.format(args.dataset, args.model, lr, batch_size, time.strftime('%m_%d_%H_%M')))

    return best_acc, best_recall, best_f1, best_epochs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--root', type=str, default='/home/hhn/hhn/data/YBUS')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='YBUS')
    parser.add_argument('--multi_GPU', type=int, default=0)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--acc', type=float, default=0.91)
    parser.add_argument('--save_path', type=str, default='checkpoint')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(args.acc, args.pretrain)

    train_data = choose_dataset(args.dataset, args.root, mode='train')
    val_data = choose_dataset(args.dataset, args.root, mode='val')

    lr_bs_search = np.random.rand(4, 5)
    for i in range(4 * 5):
        lr, batch_size = lr_bs_search[i//5, i%5]*1.5*args.lr + 0.1*args.lr, args.batch_size//(2**(i//5))
        print('{}/15'.format(i+1))
        model = eval('torchvision.models.{}'.format(args.model))(pretrained=False, num_classes=args.num_classes)
        if args.pretrain:
            pretrained_model = eval('torchvision.models.{}'.format(args.model))(pretrained=True)
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if ('fc' not in k) and ('classifier' not in k)}
            model.load_state_dict(pretrained_dict, strict=False)
        if args.multi_GPU:
            model = nn.DataParallel(model)
        best_acc, best_recall, best_f1, best_epochs = train(model, train_data, val_data, lr, batch_size, args)
        best_epochs = max(best_epochs, args.epochs)
        with open(os.path.join('log', args.dataset, args.model + '.txt'), 'a+') as f:
            f.write('{} lr={} batch_size={} best_epochs={} best_acc={} best_recall={} best_f1={}\n'.format(
                time.strftime('%m/%d %H:%M'), lr, batch_size, best_epochs, best_acc, best_recall, best_f1))
        torch.cuda.empty_cache()
