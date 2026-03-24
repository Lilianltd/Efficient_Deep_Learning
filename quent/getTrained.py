'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import argparse

from models import *
from utils import progress_bar

def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes, device=label.device).scatter_(
        1, label.view(-1, 1), 1)

def mixup(data, targets, alpha, n_classes):
    indices = torch.randperm(data.size(0), device=data.device)
    data2 = data[indices]
    targets2 = targets[indices]

    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)

    lam = torch.tensor(np.random.beta(alpha, alpha), device=data.device, dtype=data.dtype)
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
# torch.cuda.empty_cache()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
history_train_acc = []
history_test_acc = []
history_train_loss = []
history_test_loss = []

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    # transforms.RandomErasing(p=0.05, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(0,0,0)), # p=0.05 for training
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
n_classes = len(classes)

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # Assume we resume from the *last* known good file or let the user specify.
    # But usually one resumes from "ckpt.pth" which is often the best or last.
    # The default code was loading './checkpoint/ckpt.pth'. The user might have renamed/altered saving logic.
    # Let's keep loading from the default if exists, assuming user will point to correct file or default.
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    if 'history_train_acc' in checkpoint:
        history_train_acc = checkpoint['history_train_acc']
    if 'history_test_acc' in checkpoint:
        history_test_acc = checkpoint['history_test_acc']
    if 'history_train_loss' in checkpoint:
        history_train_loss = checkpoint['history_train_loss']
    if 'history_test_loss' in checkpoint:
        history_test_loss = checkpoint['history_test_loss']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    #   momentum=0, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if np.random.rand() < 0.2: #20% de chance de faire un mixup
            inputs, targets_mixed = mixup(inputs, targets, alpha=1.0, n_classes=n_classes)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets_mixed)
        else:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100.*correct/total, train_loss/(batch_idx+1)


def test(epoch, train_acc, train_loss):
    global best_acc
    global history_train_acc
    global history_test_acc
    global history_train_loss
    global history_test_loss
    
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Calculate test accuracy
    acc = 100.*correct/total
    avg_test_loss = test_loss/(batch_idx+1)
    
    # Update histories
    history_train_acc.append(train_acc)
    history_test_acc.append(acc)
    history_train_loss.append(train_loss)
    history_test_loss.append(avg_test_loss)

    # Save checkpoint at EVERY epoch
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'train_acc': train_acc,
        'loss': avg_test_loss,
        'train_loss': train_loss,
        'epoch': epoch,
        'history_train_acc': history_train_acc,
        'history_test_acc': history_test_acc,
        'history_train_loss': history_train_loss,
        'history_test_loss': history_test_loss,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    
    # Save with epoch number to keep history and allow tracing back
    # torch.save(state, './checkpoint/ckpt_epoch_{}_efficientnet.pth'.format(epoch))
    
    # Update 'ckpt.pth' to always point to the latest (for easy resume)
    # torch.save(state, './checkpoint/ckpt.pth')

    if acc > best_acc:
        # Also keep track of the absolute best model
        torch.save(state, './checkpoint/ckpt_efficientnetb0.pth')
        best_acc = acc


if __name__ == '__main__': # if nécéssaire pour éviter les problèmes de multiprocessing sur Windows
    for epoch in range(start_epoch, start_epoch+200):
        # On récupère l'accuracy retournée par train
        train_acc, train_loss = train(epoch)
        # On la passe à test
        test(epoch, train_acc, train_loss)
        scheduler.step()