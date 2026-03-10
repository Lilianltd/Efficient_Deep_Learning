"""
Prune unstructurellement un réseau pruné structurellement
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune

import matplotlib.pyplot as plt

from utils import progress_bar


def test(model, testloader, device, criterion, half=False):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if half:
                inputs, targets = inputs.to(device).half(), targets.to(device)
            else:
                inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    return acc


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

# ── Load structurally pruned model (full object, architecture changed) ────────
checkpoint_path = 'checkpoint/ckpt_efficientnetb0_structured_pruned_full.pth'
print(f'==> Loading structurally pruned model from {checkpoint_path}..')
model = torch.load(checkpoint_path, map_location='cpu')
model.eval()
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

parameters_to_prune = []
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        parameters_to_prune.append((module, 'weight'))
parameters_to_prune = tuple(parameters_to_prune)

pruning_ratio = 0.7
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=pruning_ratio,
)

sparsities = []
# print sparsity
for i in range(len(parameters_to_prune)):
    module, param_name = parameters_to_prune[i]
    print(
        "Sparsity in {}.{}: {:.2f}%".format(
            module.__class__.__name__,
            param_name,
            100. * float(torch.sum(getattr(module, param_name) == 0))
            / float(getattr(module, param_name).nelement()),
        )
    )
    sparsities.append(100. * float(torch.sum(getattr(module, param_name) == 0)) / float(getattr(module, param_name).nelement()))

# plot the sparsity distribution across layers
plt.figure(figsize=(10, 5))
plt.bar(range(len(sparsities)), sparsities)
plt.xlabel('Layer Index')
plt.ylabel('Sparsity (%)')
plt.ylim(0, 100)
plt.title('Sparsity Distribution Across Layers (Structured + Unstructured)')
plt.grid()
plt.savefig('sparsity_distribution_structured.png')

print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            sum(torch.sum(getattr(module, param_name) == 0) for module, param_name in parameters_to_prune)
        )
        / float(
            sum(getattr(module, param_name).nelement() for module, param_name in parameters_to_prune)
        )
    )
)

# ── Save ──────────────────────────────────────────────────────────────────────
# Save the full model object (architecture modified by structural pruning)
net_to_save = model.module if hasattr(model, 'module') else model
torch.save(net_to_save, 'checkpoint/ckpt_efficientnetb0_structured_unstructured_pruned_full.pth')
print('Full model saved to checkpoint/ckpt_efficientnetb0_structured_unstructured_pruned_full.pth')

# Also save dict-style checkpoint
torch.save({
    'net': net_to_save.state_dict(),
    'acc': 0,
    'epoch': 0,
    'pruning_ratio': pruning_ratio,
}, 'checkpoint/ckpt_efficientnetb0_structured_unstructured_pruned.pth')
print('Dict checkpoint saved to checkpoint/ckpt_efficientnetb0_structured_unstructured_pruned.pth')

# ── Test accuracy ─────────────────────────────────────────────────────────────
print('\n==> Preparing data for accuracy test..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()

acc = test(model, testloader, device, criterion, half=False)
print('Test accuracy with structured + unstructured pruning: %.3f%%' % acc)
