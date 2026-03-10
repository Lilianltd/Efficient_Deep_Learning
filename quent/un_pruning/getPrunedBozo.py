import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune

import matplotlib.pyplot as plt

import models
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

loaded_cpt = torch.load('checkpoint/ckpt_efficientnetb0.pth')
model = models.EfficientNetB0()
new_state_dict={
    k.replace('module.',''):v for k,v in loaded_cpt['net'].items()
} #va savoir pk tout commence par 'module.' dans le state dict
model.load_state_dict(new_state_dict)
model.eval()
# print(model)
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

parameters_to_prune = []
for module in model.modules():
    # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
    if isinstance(module, torch.nn.Conv2d):
        parameters_to_prune.append((module, 'weight'))
parameters_to_prune = tuple(parameters_to_prune)

# print(parameters_to_prune)

pruning_ratio = 0.98
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
# remettre l'échelle y entre 0 et 100
plt.ylim(0, 100)
plt.title('Sparsity Distribution Across Layers')
plt.grid()
# plt.show()
plt.savefig('sparsity_distribution.png')




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

# remove
# for module, param_name in parameters_to_prune:
#     prune.remove(module, param_name)


# sauvegarde du modèle pruné
torch.save({
    'net': model.state_dict(),
    'acc': 0, # on peut calculer l'acc du modèle pruné
    'epoch': 0,
    'pruning_ratio': pruning_ratio,
}, 'checkpoint/ckpt_efficientnetb0_pruned.pth')