"""
Prune structurellement un réseu non pruné
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from custom_utils import load_checkpoint_meta, save_checkpoint_meta

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch_pruning as tp

import models
from utils import progress_bar

PRUNING_RATIO = 0.7

# ── Device ────────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)


# ── Helpers ───────────────────────────────────────────────────────────────────
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test(model, testloader, criterion):
    model.eval()
    test_loss = 0
    correct   = 0
    total     = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100. * correct / total


if __name__ == '__main__':
    # ── Data ──────────────────────────────────────────────────────────────────
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset    = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)
    criterion  = nn.CrossEntropyLoss()

    # ── Load model ────────────────────────────────────────────────────────────
    print('==> Loading checkpoint..')
    checkpoint_path = '/homes/q23tripa/Efficient_Deep_Learning/saved_checkpoint/EfficientNetB0_9098/ckpt.pth'
    net_dict, history, ckpt = load_checkpoint_meta(checkpoint_path, device=device)
    model = models.EfficientNetB0()
    new_state_dict = {k.replace('module.', ''): v for k, v in net_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    params_before = count_parameters(model)
    print(f'\nParameters before pruning: {params_before:,}')

    print('\nAccuracy before pruning:')
    acc_before = test(model, testloader, criterion)
    print(f'Acc: {acc_before:.2f}%')

    # ── Structured pruning with torch-pruning ─────────────────────────────────
    print(f'\n==> Applying structured pruning (ratio={PRUNING_RATIO})..')

    # torch-pruning needs the model on CPU while building the dependency graph
    model.cpu()

    example_input = torch.randn(1, 3, 32, 32)

    # L2-norm importance: remove filters with the smallest L2 norm
    importance = tp.importance.MagnitudeImportance(p=2)

    # Do NOT prune the final classifier
    ignored_layers = [model.linear]

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_input,
        importance=importance,
        pruning_ratio=PRUNING_RATIO,
        ignored_layers=ignored_layers,
        global_pruning=True,    # global: same logic as unstructured global above
    )

    # One-shot pruning step (physically removes channels)
    pruner.step()

    model = model.to(device)
    params_after = count_parameters(model)
    print(f'\nParameters after  pruning: {params_after:,}')
    print(f'Reduction: {100*(1 - params_after/params_before):.1f}%  '
          f'({params_before:,} → {params_after:,})')

    # ── Test pruned model ─────────────────────────────────────────────────────
    print('\nAccuracy after structured pruning (no fine-tuning):')
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    acc_after = test(model, testloader, criterion)
    print(f'Acc: {acc_after:.2f}%')

    # ── Save ──────────────────────────────────────────────────────────────────

    net_to_save = model.module if hasattr(model, 'module') else model
    
    # Append to history
    history.append(f'sP{int(PRUNING_RATIO*100)}')
    
    # Use save_checkpoint_meta
    save_dir = '/homes/q23tripa/Efficient_Deep_Learning/quent_checkpoint'
    out_path = save_checkpoint_meta(
        model=net_to_save, 
        history=history, 
        acc=acc_after, 
        save_dir=save_dir, 
        pruning_ratio=PRUNING_RATIO, 
        params_before=params_before, 
        params_after=params_after
    )
    
    # Save the full model object so it can be reloaded without rebuilding the architecture
    save_path_full = out_path.replace('.pth', '_full.pth')
    torch.save(net_to_save, save_path_full)
    print(f'Full model object saved to {save_path_full}')
