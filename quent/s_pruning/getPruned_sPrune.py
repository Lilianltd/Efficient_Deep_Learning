"""
Prune structurellement un réseu non pruné
"""


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch_pruning as tp

import models
from utils import progress_bar

PRUNING_RATIO = 0.5

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
    loaded_cpt = torch.load('checkpoint/ckpt_efficientnetb0.pth', map_location=device)
    model = models.EfficientNetB0()
    new_state_dict = {k.replace('module.', ''): v for k, v in loaded_cpt['net'].items()}
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
    torch.save({
        'net'            : net_to_save.state_dict(),
        'acc'            : acc_after,
        'epoch'          : 0,
        'pruning_ratio'  : PRUNING_RATIO,
        'params_before'  : params_before,
        'params_after'   : params_after,
    }, 'checkpoint/ckpt_efficientnetb0_structured_pruned.pth')
    print('\nCheckpoint saved to checkpoint/ckpt_efficientnetb0_structured_pruned.pth')

    # Save the full model object so it can be reloaded without rebuilding the architecture
    torch.save(net_to_save, 'checkpoint/ckpt_efficientnetb0_structured_pruned_full.pth')
    print('Full model object saved to checkpoint/ckpt_efficientnetb0_structured_pruned_full.pth')
