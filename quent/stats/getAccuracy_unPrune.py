import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import models
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

loaded_cpt = torch.load('checkpoint/ckpt_efficientnetb0_pruned.pth')
model = models.EfficientNetB0()
new_state_dict={
    k.replace('module.',''):v for k,v in loaded_cpt['net'].items()
} #va savoir pk tout commence par 'module.' dans le state dict
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

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation()
    transforms.ToTensor(),
    # BEGIN MODIF
    transforms.RandomRotation(180),
    # transforms.BatchNorm2d(3),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
    # END MODIF
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

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# TEST DE BASE
# acc = test(model, testloader, device, criterion)
# print('Test accuracy: %.3f%%' % acc)

# TEST AVEC HALF PRECISION
# halved = model.half()
# acc = test(halved, testloader, device, criterion, half=True)
# print('Test accuracy with half precision: %.3f%%' % acc)

# TEST AVEC PRUNING
acc = test(model, testloader, device, criterion, half=False)
print('Test accuracy with pruning: %.3f%%' % acc)