'''Fine-tune a pruned (structurally, unstructurally, or both) model'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from custom_utils import load_checkpoint_meta, save_checkpoint_meta
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune

import torchvision
import torchvision.transforms as transforms
import numpy as np

import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Fine-Tuning Pruned Model')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate') # lower than usual
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--structured', action='store_true', help='fine-tune a purely structured pruned model')
parser.add_argument('--mixed', action='store_true', help='fine-tune a structured + unstructured pruned model')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
best_acc = 0  
start_epoch = 0
n_epochs = 9

# ── Data ──────────────────────────────────────────────────────────────────────
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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

save_path = None
parameters_to_prune = []

# ── Chargement du Modèle ──────────────────────────────────────────────────────

if args.mixed:
    # ── Structured + Unstructured pruned model ──
    checkpoint_path = '/homes/q23tripa/Efficient_Deep_Learning/quent_checkpoint/EfficientNet_sP65F10unP70_full.pth'
    print(f'==> Loading mixed pruned model from {checkpoint_path}..')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing {checkpoint_path}")
    
    try:
        net_dict, history, ckpt = load_checkpoint_meta(checkpoint_path.replace('_full.pth', '.pth'), device=device)
    except:
        history = []
        
    # On charge l'objet complet car l'architecture a été modifiée par le pruning structuré
    net = torch.load(checkpoint_path, map_location=device)
    net = net.to(device)
    
    # On détecte dynamiquement les couches qui ont subi le pruning non-structuré
    for module in net.modules():
        if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
            parameters_to_prune.append((module, 'weight'))
            
    best_acc = 0
    print(f'Mixed pruned model loaded. Found {len(parameters_to_prune)} layers with unstructured pruning masks.')

elif args.structured:
    # ── Purely Structured pruned model ──
    checkpoint_path = '/homes/q23tripa/Efficient_Deep_Learning/quent_checkpoint/EfficientNet_sP65F10unP70_full.pth'
    print(f'==> Loading structured pruned model from {checkpoint_path}..')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing {checkpoint_path}")
    
    try:
        net_dict, history, ckpt = load_checkpoint_meta(checkpoint_path.replace('_full.pth', '.pth'), device=device)
    except:
        history = []
        
    net = torch.load(checkpoint_path, map_location=device)
    net = net.to(device)
    best_acc = 0
    print('Structured pruned model loaded.')

else:
    # ── Purely Unstructured pruned model ──
    net = EfficientNetB0()
    net = net.to(device)

    print('==> Applying pruning structure..')
    for module in net.modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.7, 
    )

    print('==> Loading pruned checkpoint..')
    checkpoint_path = '/homes/q23tripa/Efficient_Deep_Learning/quent_checkpoint/EfficientNet_unP10.pth'
    if os.path.exists(checkpoint_path):
        net_dict, history, checkpoint = load_checkpoint_meta(checkpoint_path, device=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in net_dict.items()}
        net.load_state_dict(new_state_dict)
        if 'acc' in checkpoint:
            best_acc = checkpoint['acc']
            print(f"Loaded checkpoint '{checkpoint_path}' (acc: {best_acc})")
    else:
        print(f"Error: checkpoint '{checkpoint_path}' not found!")
        history = []

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
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

def test(epoch, state):
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

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving Best Accuracy..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        best_acc = acc
    return state

if __name__ == '__main__':
    state = {'acc': 0}
    for epoch in range(start_epoch, start_epoch+n_epochs):
        train(epoch)

        # On affiche la sparsité si le modèle contient des masques (non-structuré ou mixte)
        if not args.structured:
            total = sum(p.numel() for n, p in net.named_buffers() if 'mask' in n)
            if total > 0:
                zeros = sum((p == 0).sum().item() for n, p in net.named_buffers() if 'mask' in n)
                print(f"Sparsity after epoch: {100.*zeros/total:.1f}%")

        state = test(epoch, state)
        scheduler.step()

    # Nettoyage des hooks de pruning pour les modèles non-structurés ou mixtes
    if not args.structured and len(parameters_to_prune) > 0:
        print("\n==> Removing pruning hooks making sparsity permanent...")
        for module, param_name in parameters_to_prune:
            # Sécurité anti-bug PyTorch : on vérifie que le hook est bien là avant de le retirer
            if hasattr(module, 'weight_orig'):
                prune.remove(module, param_name)

    net_to_save = net.module if hasattr(net, 'module') else net
    
    history.append(f"F{n_epochs}") 
    
    out_path = save_checkpoint_meta(
        model=net_to_save,
        history=history,
        acc=state['acc'],
        save_dir='/homes/q23tripa/Efficient_Deep_Learning/quent_checkpoint',
        finetune_epochs=n_epochs
    )
    
    # Pour 'mixed' ou 'structured', l'architecture est custom, on doit sauvegarder l'objet complet
    if args.structured or args.mixed:
        save_path_full = out_path.replace('.pth', '_full.pth')
        torch.save(net_to_save, save_path_full)
        print(f'Final full model saved to {save_path_full}')

def count_params_and_zeros(m):
    total = 0
    zeros = 0
    for module in m.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            if hasattr(module, 'weight') and getattr(module, 'weight') is not None:
                w = getattr(module, 'weight')
                total += w.numel()
                zeros += torch.sum(w == 0).item()
            if hasattr(module, 'bias') and getattr(module, 'bias') is not None:
                b = getattr(module, 'bias')
                total += b.numel()
                zeros += torch.sum(b == 0).item()
    return total, zeros

total_after, zeros_after = count_params_and_zeros(net_to_save)
print(f"\nTotal params: {total_after}, Zeros: {zeros_after} ({100*zeros_after/total_after:.2f}%)")