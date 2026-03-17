"""
Fine-tuning QAT (Quantization-Aware Training) sur x-bits
pour un modèle ayant subi un pruning structuré ET non-structuré.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from custom_utils import load_checkpoint_meta, save_checkpoint_meta

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune
import torch.ao.quantization as quant

from utils import progress_bar

# ── Paramètres QAT ────────────────────────────────────────────────────────────
X_BITS = 4  # Remplace par le nombre de bits souhaité pour les poids
EPOCHS = 5
LR = 1e-5   # Learning rate très faible recommandé pour le QAT

def test(model, testloader, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100. * correct / total

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    # ── 1. Data ───────────────────────────────────────────────────────────────
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

    criterion = nn.CrossEntropyLoss()

    # ── 2. Load Model ─────────────────────────────────────────────────────────
    # Utilise le modèle issu de getUnPruned_sPrune.py
    checkpoint_path = 'checkpoint/ckpt_efficientnetb0_structured_pruned_full.pth' # A ADAPTER
    print(f'==> Loading unstructured pruned model from {checkpoint_path}..')
    
    try:
        net_dict, history, ckpt = load_checkpoint_meta(checkpoint_path.replace('_full.pth', '.pth'), device='cpu')
    except FileNotFoundError:
        history = []
        
    model = torch.load(checkpoint_path, map_location=device)
    
    # Si le modèle était en DataParallel, on le "déballe" pour éviter les bugs avec PyTorch QAT
    if hasattr(model, 'module'):
        model = model.module

    # ── 3. Rendre le Pruning Permanent & Extraire les masques ─────────────────
    print('==> Making unstructured pruning permanent and saving masks..')
    masks = {}
    for name, module in model.named_modules():
        # Si le module a des hooks de pruning en attente (weight_orig)
        if prune.is_pruned(module):
            prune.remove(module, 'weight') # Fusionne weight_orig et le masque dans 'weight'
        
        # On sauvegarde les zéros comme un masque binaire pour le QAT
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            masks[name] = (module.weight != 0).float().to(device)
            
    # Vérification rapide
    zero_weights = sum((m == 0).sum().item() for m in masks.values())
    print(f"    Sparsity mask saved. Total zeroed weights: {zero_weights:,}")

    # ── 4. Configuration QAT sur X bits ───────────────────────────────────────
    print(f'==> Preparing QAT for {X_BITS}-bit weights..')
    model.train()
    
    # Calcul des bornes pour x bits (symétrique)
    qmin = -(2 ** (X_BITS - 1))
    qmax = (2 ** (X_BITS - 1)) - 1
    
    # Configuration FakeQuantize personnalisée pour les poids
    weight_fq = quant.FakeQuantize.with_args(
        observer=quant.MinMaxObserver,
        quant_min=qmin,
        quant_max=qmax,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False
    )
    
    # On utilise la configuration par défaut pour les activations (8-bits), 
    # et notre config personnalisée pour les poids
    qconfig = quant.QConfig(
        activation=quant.FakeQuantize.with_args(observer=quant.MovingAverageMinMaxObserver, quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=True),
        weight=weight_fq
    )
    
    model.qconfig = qconfig
    # Insère les noeuds FakeQuantize dans le modèle
    model_qat = quant.prepare_qat(model, inplace=False)
    model_qat = model_qat.to(device)

    # ── 5. Fine-Tuning QAT (Training Loop) ────────────────────────────────────
    print('==> Starting QAT Fine-tuning..')
    optimizer = optim.Adam(model_qat.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        print(f'\nEpoch: {epoch+1}/{EPOCHS}')
        model_qat.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model_qat(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # --- CRITIQUE : MAINTIEN DE LA SPARSITÉ ---
            # Après la mise à jour des poids, on force les poids prunés à rester à zéro
            with torch.no_grad():
                for name, module in model_qat.named_modules():
                    if name in masks:
                        # Les modules originaux (ex: conv1) gardent leur nom même enveloppés par qat
                        module.weight.data.mul_(masks[name])
            # ------------------------------------------

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%%'
                         % (train_loss/(batch_idx+1), 100.*correct/total))

    # ── 6. Test & Save ────────────────────────────────────────────────────────
    print('\n==> Testing QAT model..')
    acc = test(model_qat, testloader, device, criterion)
    print(f'Test accuracy after {X_BITS}-bit QAT: {acc:.3f}%')

    history.append(f'QAT{X_BITS}b')
    
    net_to_save = model_qat
    save_dir = '/homes/q23tripa/Efficient_Deep_Learning/quent_checkpoint'
    out_path = save_checkpoint_meta(
        model=net_to_save,
        history=history,
        acc=acc,
        save_dir=save_dir
    )
    save_path_full = out_path.replace('.pth', '_full.pth')
    torch.save(net_to_save, save_path_full)
    print(f'Full QAT model saved to {save_path_full}')