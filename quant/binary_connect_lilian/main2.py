import torch
from train_routine_lilian.main import mixup_criterion, mixup_data, load_data
'''Train CIFAR10 with PyTorch.'''
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchinfo
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from models import *


def train(net, epoch, trainloader, optimizer, criterion, alpha=1.0):
    print('\nEpoch bc: %d' % epoch)
    net.model.train()

    train_loss = 0
    correct = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, use_cuda=(device == 'cuda'))

        optimizer.zero_grad()

        net.binarization()
        
        outputs = net.model(inputs)

        # ---- MIXUP LOSS ----
        loss = mixup_criterion(
            criterion, outputs, targets_a, targets_b, lam
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # ---- MIXUP ACCURACY (weighted) ----
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (
            lam * predicted.eq(targets_a).sum().item()
            + (1 - lam) * predicted.eq(targets_b).sum().item()
        )

        net.restore()
        
        optimizer.step()
        net.clip()

    avg_loss = train_loss / len(trainloader)
    acc = 100. * correct / total

    return avg_loss, acc

def main(net, parameter, subfolder):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0

    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/{subfolder}")

    net.model.to(device)
    net_cuda = net.model
    if device == 'cuda':
        net_cuda = torch.nn.DataParallel(net_cuda)
        cudnn.benchmark = True
    trainloader, testloader, classes = load_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net_cuda.parameters(),
        parameter["lr"],
        momentum=parameter["momentum"],
        weight_decay=parameter["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    count = 0

    for epoch in range(start_epoch, start_epoch + 300):

        train_loss, train_acc = train(
            net, epoch, trainloader, optimizer, criterion, alpha=parameter["alpha"]
        )
        test_loss, test_acc, best_acc, flag_improve = test(
            net, epoch, testloader, subfolder, criterion, best_acc
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        scheduler.step()

    trainable_params = sum(p.numel() for p in net_cuda.parameters() if p.requires_grad)
    writer.add_scalar("Accuracy/best", best_acc, trainable_params)

    writer.close()


def test(net, epoch, testloader, subfolder, criterion, best_acc):
    net.model.eval()

    test_loss = 0
    correct = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        net.binarization()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)    
            outputs = net.model(inputs)
            loss = criterion(outputs.float(), targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        net.restore()
    avg_loss = test_loss / len(testloader)
    acc = 100. * correct / total
    flag_improve = False

    if acc > best_acc:
        best_acc = acc
        print(f'Saving.. best_acc : {best_acc}')

        state = {
            'net': net.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        os.makedirs(f'./checkpoint/{subfolder}/', exist_ok=True)
        torch.save(state, f'./checkpoint/{subfolder}/ckpt.pth')

        best_acc = acc
        flag_improve = True

    return avg_loss, acc, best_acc, flag_improve