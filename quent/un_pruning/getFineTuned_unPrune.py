'''Fine-tune a pruned DenseNet121.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Fine-Tuning Pruned Model')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate') # Reduced LR for fine-tuning
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--structured', action='store_true', help='fine-tune a structured pruned model')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
best_acc = 0  
start_epoch = 0

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
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

if args.structured:
    # ── Structured pruned model: architecture is physically smaller, load full object ──
    checkpoint_path = 'checkpoint/ckpt_efficientnetb0_structured_pruned_full.pth'
    print(f'==> Loading structured pruned model from {checkpoint_path}..')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Run getStructuredPrunedBozo.py first to generate {checkpoint_path}")
    net = torch.load(checkpoint_path, map_location=device)
    net = net.to(device)
    parameters_to_prune = []  # no unstructured masks to remove later
    best_acc = 0
    print('Structured pruned model loaded.')
else:
    # ── Unstructured pruned model: same architecture, masks applied ───────────
    net = EfficientNetB0() # Instantiate the model architecture first
    net = net.to(device)

    # 1. Apply Pruning Structure BEFORE loading state_dict
    print('==> Applying pruning structure..')
    parameters_to_prune = []
    for module in net.modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.7, # will be overridden by the loaded state_dict
    )

    # 2. Load the unstructured pruned checkpoint
    print('==> Loading pruned checkpoint..')
    checkpoint_path = 'checkpoint/ckpt_efficientnetb0_pruned.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if 'net' in checkpoint:
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
            net.load_state_dict(new_state_dict)
            best_acc = checkpoint['acc']
            print(f"Loaded checkpoint '{checkpoint_path}' (acc: {best_acc})")
        else:
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
            net.load_state_dict(new_state_dict)
            print(f"Loaded state_dict from '{checkpoint_path}'")
    else:
        print(f"Error: checkpoint '{checkpoint_path}' not found!")

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# Use a smaller learning rate for fine-tuning
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
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

def test(epoch):
    global best_acc
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

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_name = 'ckpt_efficientnetb0_structured_pruned_finetuned.pth' if args.structured else 'ckpt_efficientnetb0_pruned_finetuned.pth'
        torch.save(state, f'./checkpoint/{save_name}')
        best_acc = acc
    return state

if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch+1): # Fine-tune for 20 epochs
        train(epoch)

        if not args.structured:
            total = sum(p.numel() for n, p in net.named_buffers() if 'mask' in n)
            zeros = sum((p == 0).sum().item() for n, p in net.named_buffers() if 'mask' in n)
            print(f"Sparsity after epoch: {100.*zeros/total:.1f}%")

        state = test(epoch)
        scheduler.step()

    if not args.structured:
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

    # Re-capture state_dict AFTER prune.remove() so weights are clean
    net_to_save = net.module if hasattr(net, 'module') else net
    state['net'] = net_to_save.state_dict()
    save_name = 'ckpt_efficientnetb0_structured_pruned_finetuned.pth' if args.structured else 'ckpt_efficientnetb0_pruned_finetuned.pth'
    torch.save(state, f'./checkpoint/{save_name}')
    print(f'Final checkpoint saved to checkpoint/{save_name}')



# '''Fine-tune a pruned DenseNet121.'''
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# import torch.nn.utils.prune as prune

# import torchvision
# import torchvision.transforms as transforms
# import numpy as np

# import os
# import argparse

# from models import *
# from utils import progress_bar

# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Fine-Tuning Pruned Model')
# parser.add_argument('--lr', default=0.001, type=float, help='learning rate') # Reduced LR for fine-tuning
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# args = parser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using device:', device)
# best_acc = 0  
# start_epoch = 0

# # Data
# print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=100, shuffle=False, num_workers=2)

# # Model
# net = EfficientNetB0()
# net = net.to(device)


# # loaded_cpt = torch.load('checkpoint/ckpt_lilian.pth')
# # net = DenseNet121()
# # new_state_dict={
# #     k.replace('module.',''):v for k,v in loaded_cpt['net'].items()
# # } #va savoir pk tout commence par 'module.' dans le state dict
# # net.load_state_dict(new_state_dict)
# # if device == 'cuda':
# #     net = torch.nn.DataParallel(net)
# #     cudnn.benchmark = True

# # 1. Apply Pruning Structure BEFORE loading state_dict
# # PyTorch needs the pruning structure (masks, ref to _orig weights) to be present
# # to load a state_dict that contains pruning buffers.
# print('==> Applying pruning structure..')
# parameters_to_prune = []
# for module in net.modules():
#     if isinstance(module, torch.nn.Conv2d):
#         parameters_to_prune.append((module, 'weight'))

# # Apply identity pruning just to initialize the buffers/hooks
# prune.global_unstructured(
#     parameters_to_prune,
#     pruning_method=prune.L1Unstructured,
#     amount=0.7, # will be overridden by the loaded state_dict, so DONT CARE + DIDNT ASK + RATIO
# )

# # 2. load le pruned checkpoint
# print('==> Loading pruned checkpoint..')
# checkpoint_path = 'checkpoint/ckpt_efficientnetb0_pruned.pth' 
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path)
#     if 'net' in checkpoint:

#         new_state_dict={
#             k.replace('module.',''):v for k,v in checkpoint['net'].items()
#         } #va savoir pk tout commence par 'module.' dans le state dict
#         net.load_state_dict(new_state_dict)

#         # net.load_state_dict(checkpoint['net'])
#         best_acc = checkpoint['acc']
#         print(f"Loaded checkpoint '{checkpoint_path}' (acc: {best_acc})")
#     else:
#         # Fallback if checkpoint is just state_dict
#         new_state_dict={
#             k.replace('module.',''):v for k,v in checkpoint['net'].items()
#         } #va savoir pk tout commence par 'module.' dans le state dict
#         net.load_state_dict(new_state_dict)
#         # net.load_state_dict(checkpoint)
#         print(f"Loaded state_dict from '{checkpoint_path}'")
# else:
#     print(f"Error: checkpoint '{checkpoint_path}' not found!")
#     # exit() # Uncomment to enforce checkpoint existence

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# criterion = nn.CrossEntropyLoss()
# # Use a smaller learning rate for fine-tuning
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#         progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# def test(epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#             progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt_efficientnetb0_pruned_finetuned.pth')
#         best_acc = acc
#     return state

# for epoch in range(start_epoch, start_epoch+2): # Fine-tune for 2 epochs
#     train(epoch)
#     state = test(epoch)
#     scheduler.step()

# for module, param_name in parameters_to_prune:
#     prune.remove(module, param_name)
# torch.save(state, './checkpoint/ckpt_efficientnetb0_pruned_finetuned.pth')
