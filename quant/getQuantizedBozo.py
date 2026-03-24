import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans
from models import *
from pruning.pruning_lilian.utils import load_and_make_permanent
from custom_utils import score
from train_routine_lilian.main import test

QUANTIZATION_BITS = 8
N_LEVELS = 2**QUANTIZATION_BITS - 1 # -1 pour que ce soit symmetrique
N_POS    = N_LEVELS // 2 # 


def find_quantization_levels(model):
    """
    Run k-means on |non-zero weights| to find N_POS positive centroids.
    Returns the full symmetric set of 15 levels: -c7..-c1, 0, c1..c7.
    """
    # nb : on triche et on fait ça que sur la moitié positive puis on miroir
    # sur la partie négative, du coup on force une symmétrie
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
                # Vectorised nearest-level: broadcast (N, 1) vs (15,)
                indices = (w_flat.unsqueeze(-1) - levels_t).abs().argmin(dim=-1)
                param.data = levels_t[indices].reshape(param.shape).to(param.device)

MODEL_PATH = '/homes/l22letar/EDL/pytorch-cifar/saved_checkpoint/MobileNetV2_custom_0.4_0.01_0.0005_0.9/ckpt.pth'
ModelClass = MobileNetV2_Custom
model_args = {"width_mult":0.5}
model_name = "Mobilnet"

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Load pruned+finetuned checkpoint ──────────────────────────────────────
    model = load_and_make_permanent(MODEL_PATH, ModelClass, model_args)
    model.to("cuda").eval().half()
    print(model)
    print(score(model, 0.1, 0.9*0.5, 8, 16))

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
    plt.savefig("test.png")

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

    print("==> Testing quantized model..")
    avg_loss, acc, best_acc, flag_improve = test(model.half(), 0, testloader, "", criterion, 0, half=True)
    print(f"\n  Accuracy  (4-bit quantized + pruned): {best_acc:.2f}%")
    print(f"  Test loss                           : {avg_loss:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path   = './checkpoint/ckpt_densenet_lilian_pruned_quantized.pth'
    net_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
        'net'     : net_to_save.state_dict(),
        'acc'     : acc,
        'levels'  : levels.tolist(),
        'bits'    : QUANTIZATION_BITS,
        'n_levels': N_LEVELS,
    }, out_path)
    print(f"==> Quantized model saved to {out_path}")
    

