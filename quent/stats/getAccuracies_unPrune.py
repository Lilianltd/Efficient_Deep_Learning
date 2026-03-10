import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt

import models
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

loaded_cpt = torch.load('checkpoint/ckpt_efficientnetb0.pth')
prunedModels = [models.EfficientNetB() for shrek in range(30)]
new_state_dict={
    k.replace('module.',''):v for k,v in loaded_cpt['net'].items()
} #va savoir pk tout commence par 'module.' dans le state dict
for model in prunedModels:
    model.load_state_dict(new_state_dict)
    model.eval()
# print(model)



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

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()



accuracies = []
pruning_ratios = []
k = 0.0
for model in prunedModels:
    if k < 0.6:
        k += 0.1
    elif k < 0.90:
        k += 0.025
    elif k < 0.99:
        k += 0.01
    elif k > 0.999:
        break
    
    if k > 0.999: # la fameuse A2F
        break
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

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=k,
    )

    # remove
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    model.eval()
    acc = test(model, testloader, device, criterion)
    accuracies.append(acc)
    pruning_ratios.append(k)
    print(f'Pruning ratio: {k:.2f}, Accuracy: {acc:.2f}%')
    
plt.plot(pruning_ratios, accuracies, marker='o')
plt.title('Accuracy vs Pruning Ratio')
plt.xlabel('Pruning Ratio')
plt.ylabel('Accuracy (%)')
plt.axhline(y=90, color='r', linestyle='--', label='90% Accuracy') # put a line at 90% accuracy
plt.legend()
plt.grid()
plt.savefig('accuracy_vs_pruning_ratio.png')
# plt.show() # çA MARCHE PAS EN SSH RAAAAA


# sauvegarde du modèle pruné
# for i, model in enumerate(prunedModels):
#     torch.save({
#         'net': model.state_dict(),
#         'acc': 0, # on peut calculer l'acc du modèle pruné
#         'epoch': 0,
#     }, f'checkpoint/ckpt_lilian_pruned_{i}.pth')