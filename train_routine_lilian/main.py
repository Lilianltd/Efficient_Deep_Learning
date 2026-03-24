'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

from models import *
from train_routine_lilian.utils import load_data, mixup_criterion, mixup_data

def main(net, parameter, subfolder, num_run=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0

    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/{subfolder}")

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    trainloader, testloader, classes = load_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        parameter["lr"],
        momentum=parameter["momentum"],
        weight_decay=parameter["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_run)
    for epoch in range(start_epoch, start_epoch + num_run):

        train_loss, train_acc = train(
            net, epoch, trainloader, optimizer, criterion, alpha=parameter["alpha"]
        )
        test_loss, test_acc, best_acc, _ = test(
            net, epoch, testloader, subfolder, criterion, best_acc
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        scheduler.step()

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    writer.add_scalar("Accuracy/best", best_acc, trainable_params)

    writer.close()

# Training
def train(net, epoch, trainloader, optimizer, criterion, alpha=1.0):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # ---- MIXUP START ----
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, alpha, use_cuda=(device == 'cuda')
        )
        # ---- MIXUP END ----

        optimizer.zero_grad()

        outputs = net(inputs)

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

    avg_loss = train_loss / len(trainloader)
    acc = 100. * correct / total

    return avg_loss, acc

def test(net, epoch, testloader, subfolder, criterion, best_acc, half=None):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if half:
                inputs = inputs.to(device).half()
            else:
                inputs = inputs.to(device)
            targets = targets.to(device)    
            outputs = net(inputs)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs.float(), targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / len(testloader)
    acc = 100. * correct / total
    flag_improve = False

    if acc > best_acc:
        best_acc = acc
        print(f'Saving.. best_acc : {best_acc}')

        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        os.makedirs(f'./checkpoint/{subfolder}/', exist_ok=True)
        torch.save(state, f'./checkpoint/{subfolder}/ckpt.pth')

        best_acc = acc
        flag_improve = True

    return avg_loss, acc, best_acc, flag_improve
