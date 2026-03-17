"""
Quantifie en poids un modèle pruné structurellement puis non-structurellement
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
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans

QUANTIZATION_BITS = 8
N_LEVELS = 2**QUANTIZATION_BITS - 1  # -1 pour que ce soit symétrique
N_POS    = N_LEVELS // 2


def find_quantization_levels(model):
    """
    Run k-means on |non-zero weights| to find N_POS positive centroids.
    Returns the full symmetric set of levels: -cN..-c1, 0, c1..cN.
    """
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            all_weights.extend(param.detach().cpu().numpy().flatten())

    all_weights = np.array(all_weights)
    nonzero_abs = np.abs(all_weights[all_weights != 0])

    total = len(all_weights)
    nz    = len(nonzero_abs)
    print(f"  Total weights: {total:,}  |  Non-zero: {nz:,} ({100*nz/total:.1f}%)")

    kmeans = KMeans(n_clusters=N_POS, random_state=0, n_init=10)
    kmeans.fit(nonzero_abs.reshape(-1, 1))
    print(f"  K-means inertia (sum of squared distances to nearest centroid): {kmeans.inertia_:.2f}")
    pos_levels = np.sort(kmeans.cluster_centers_.flatten())   # ascending

    # Symmetric: negative mirror + 0 + positive
    levels = np.concatenate([-pos_levels[::-1], [0.0], pos_levels])
    return levels, all_weights


def quantize_model(model, levels):
    """Replace every conv/linear weight tensor with its nearest quantization level (in-place)."""
    levels_t = torch.tensor(levels, dtype=torch.float32)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                w_flat  = param.data.cpu().flatten()
                indices = (w_flat.unsqueeze(-1) - levels_t).abs().argmin(dim=-1)
                param.data = levels_t[indices].reshape(param.shape).to(param.device)


def test(model, testloader, device, criterion):
    model.eval()
    test_loss = 0
    correct   = 0
    total     = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total, test_loss / len(testloader)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Load structurally pruned model (full object, architecture changed) ────
    checkpoint_path = '/homes/q23tripa/Efficient_Deep_Learning/quent_checkpoint/EfficientNet_sPXXF1unPXXF1_full.pth'
    print(f"==> Loading structurally pruned model from {checkpoint_path}..")
    
    try:
        net_dict, history, ckpt = load_checkpoint_meta(checkpoint_path.replace('_full.pth', '.pth'), device=device)
    except:
        history = []
        
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Structural pruning changes the architecture, so we load the full model object
    model = torch.load(checkpoint_path, map_location=device)
    model = model.to(device)

    # ── Find quantization levels via k-means ──────────────────────────────────
    print(f"==> Computing {N_LEVELS} symmetric quantization levels "
          f"(k-means, k={N_POS}, on |non-zero weights|)..")
    levels, all_weights_before = find_quantization_levels(model)
    print(f"  Levels ({N_LEVELS}): {np.round(levels, 5)}")

    # ── Plot: distribution before + level markers ─────────────────────────────
    plt.figure(figsize=(13, 5))

    plt.subplot(1, 2, 1)
    plt.hist(all_weights_before,
             bins=300, alpha=0.75, color='steelblue', edgecolor='none')
    for lv in levels:
        plt.axvline(lv, color='red', linewidth=0.8, alpha=0.8)
    plt.title('Before quantization  (red = levels)')
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # ── Quantize ──────────────────────────────────────────────────────────────
    print("==> Quantizing model weights..")
    quantize_model(model, levels)

    # Collect quantized weights for plot & verification
    all_weights_after = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            all_weights_after.extend(param.detach().cpu().numpy().flatten())
    all_weights_after = np.array(all_weights_after)

    unique_vals = np.unique(all_weights_after)
    print(f"  Unique values in quantized model: {len(unique_vals)}  (expected ≤ {N_LEVELS})")

    plt.subplot(1, 2, 2)
    plt.hist(all_weights_after, bins=len(unique_vals),
             alpha=0.75, color='darkorange', edgecolor='black')
    plt.title('After quantization')
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('wquantization_structured_histograms.png')

    # ── Evaluate accuracy ─────────────────────────────────────────────────────
    print("==> Preparing data..")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset    = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    print("==> Testing quantized structurally-pruned model..")
    acc, loss_val = test(model, testloader, device, criterion)
    print(f"\n  Accuracy  ({QUANTIZATION_BITS}-bit quantized + structured pruned): {acc:.2f}%")
    print(f"  Test loss                                          : {loss_val:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    net_to_save = model.module if hasattr(model, 'module') else model

    history.append(f'Q{QUANTIZATION_BITS}')

    out_path_dict = save_checkpoint_meta(
        model=net_to_save,
        history=history,
        acc=acc,
        save_dir='/homes/q23tripa/Efficient_Deep_Learning/quent_checkpoint',
        levels=levels.tolist(),
        bits=QUANTIZATION_BITS,
        n_levels=N_LEVELS
    )

    out_path = out_path_dict.replace('.pth', '_full.pth')
    torch.save(net_to_save, out_path)
    print(f"==> Full quantized model saved to {out_path}")
