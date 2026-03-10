import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from models import *
from train_routine_lilian.main import test
from train_routine_lilian.utils import load_data, mixup_criterion, mixup_data

def main(net, teacher_net, parameter, subfolder, num_run=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0

    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/{subfolder}")

    net = net.to(device)
    teacher_net = teacher_net.to(device)
    
    # Put teacher in eval mode (no gradients needed)
    teacher_net.eval()

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        teacher_net = torch.nn.DataParallel(teacher_net)
        cudnn.benchmark = True

    trainloader, testloader, classes = load_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        parameter["lr"],
        momentum=parameter["momentum"],
        weight_decay=parameter["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_run)
    
    # KD Hyperparameters (can be passed via `parameter` dict)
    temperature = parameter.get("temperature", 4.0)
    kd_weight = parameter.get("kd_weight", 0.5)

    for epoch in range(start_epoch, start_epoch + num_run):

        train_loss, train_acc = train(
            net, teacher_net, epoch, trainloader, optimizer, criterion, 
            alpha=parameter["alpha"], temperature=temperature, kd_weight=kd_weight
        )
        test_loss, test_acc, best_acc, _ = test(
            net, epoch, testloader, subfolder, criterion, best_acc
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        scheduler.step()

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    writer.add_scalar("Accuracy/best", best_acc, trainable_params)

    writer.close()

# Training
def train(net, teacher_net, epoch, trainloader, optimizer, criterion, alpha=1.0, temperature=4.0, kd_weight=0.5):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # ---- MIXUP START ----
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, alpha, use_cuda=(device == 'cuda')
        )
        # ---- MIXUP END ----

        optimizer.zero_grad()

        # Student logits
        outputs = net(inputs)
        
        # Teacher logits (no gradients required)
        with torch.no_grad():
            teacher_outputs = teacher_net(inputs)

        # ---- ORIGINAL MIXUP LOSS (Hard Targets) ----
        loss_ce = mixup_criterion(
            criterion, outputs, targets_a, targets_b, lam
        )
        
        # ---- KNOWLEDGE DISTILLATION LOSS (Soft Targets) ----
        # KL Divergence between softened student and teacher probabilities
        loss_kd = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(outputs / temperature, dim=1),
            F.softmax(teacher_outputs / temperature, dim=1)
        ) * (temperature ** 2)
        
        # Combine losses
        loss = (1. - kd_weight) * loss_ce + kd_weight * loss_kd

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # ---- MIXUP ACCURACY (weighted) ----
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (
            lam * predicted.eq(targets_a).sum().item()
            + (1 - lam) * predicted.eq(targets_b).sum().item()
        )

    avg_loss = train_loss / len(trainloader)
    acc = 100. * correct / total

    return avg_loss, acc
