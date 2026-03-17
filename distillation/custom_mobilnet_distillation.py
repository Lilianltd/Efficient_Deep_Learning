import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from models import *
from train_routine_lilian.main import test
from train_routine_lilian.utils import load_data, mixup_criterion, mixup_data

import os
import torch

def main(net, teacher_net, parameter, subfolder, num_run=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0
    net = net.to(device)
    writer = SummaryWriter(log_dir=f"runs/{subfolder}")
    
    wrapper = DistillationWrapper(
        net, 
        teacher_net, 
        dummy_input_size=(1, 3, 32, 32)
    ).to(device)

    # 2. Wrap the wrapper in DataParallel (not the individual models)
    if device == 'cuda':
        wrapper = torch.nn.DataParallel(wrapper)
        cudnn.benchmark = True

    trainloader, testloader, classes = load_data()

    criterion = nn.CrossEntropyLoss()
    
    # 3. CRITICAL: Optimize the wrapper's parameters (Student + Adapters)
    optimizer = optim.SGD(
        wrapper.parameters(), 
        parameter["lr"],
        momentum=parameter["momentum"],
        weight_decay=parameter["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_run)
    
    temperature = parameter.get("temperature", 4.0)
    kd_weight = parameter.get("kd_weight", 0.5)
    beta = parameter.get("beta", 1.0) # Weight for intermediate feature loss

    for epoch in range(start_epoch, start_epoch + num_run):

        train_loss, train_acc = train(
            wrapper, epoch, trainloader, optimizer, criterion, 
            alpha=parameter["alpha"], temperature=temperature, 
            kd_weight=kd_weight, beta=beta
        )
        
        # Extract the student model to test it normally
        student_net = wrapper.module.student if isinstance(wrapper, nn.DataParallel) else wrapper.student
        
        test_loss, test_acc, best_acc, _ = test(
            student_net, epoch, testloader, subfolder, criterion, best_acc
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        scheduler.step()

    student_net = wrapper.module.student if isinstance(wrapper, nn.DataParallel) else wrapper.student
    trainable_params = sum(p.numel() for p in student_net.parameters() if p.requires_grad)
    writer.add_scalar("Accuracy/best", best_acc, trainable_params)

    writer.close()

def train(wrapper, epoch, trainloader, optimizer, criterion, alpha=1.0, temperature=4.0, kd_weight=0.5, beta=1.0):
    print('\nEpoch: %d' % epoch)
    wrapper.train()

    train_loss = 0
    correct = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, alpha, use_cuda=(device == 'cuda')
        )

        optimizer.zero_grad()

        s_logits, t_logits, s_feats, t_feats = wrapper(inputs)
        loss_ce = mixup_criterion(
            criterion, s_logits, targets_a, targets_b, lam
        )
        
        loss_kd = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(s_logits / temperature, dim=1),
            F.softmax(t_logits / temperature, dim=1)
        ) * (temperature ** 2)
        
        feat_loss = 0
        
        adapters = wrapper.module.adapters if isinstance(wrapper, nn.DataParallel) else wrapper.adapters
        
        for i, adapter in enumerate(adapters):
            projected_s = adapter(s_feats[i])
            feat_loss += F.mse_loss(projected_s, t_feats[i])
        
        # 5. Combine total loss
        loss = (1. - kd_weight) * loss_ce + (kd_weight * loss_kd) + (beta * feat_loss)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = s_logits.max(1)
        total += targets.size(0)
        correct += (
            lam * predicted.eq(targets_a).sum().item()
            + (1 - lam) * predicted.eq(targets_b).sum().item()
        )

    avg_loss = train_loss / len(trainloader)
    acc = 100. * correct / total

    return avg_loss, acc

class DistillationWrapper(nn.Module):
    def __init__(self, student, teacher, dummy_input_size=(1, 3, 32, 32)):
        super().__init__()
        self.student = student
        self.teacher = teacher
        
        # 1. Geler complètement le professeur
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # 2. -- DÉTECTION AUTOMATIQUE DES DIMENSIONS --
        # On trouve sur quel appareil (CPU/GPU) le modèle se trouve
        device = next(student.parameters()).device
        
        # On crée un faux tenseur d'entrée (B, C, H, W)
        dummy_input = torch.randn(*dummy_input_size).to(device)
        
        # On fait une passe à vide sans calculer les gradients
        with torch.no_grad():
            self.student.eval() # Temporairement en eval
            _, s_feats = self.student(dummy_input)
            _, t_feats = self.teacher(dummy_input)
            self.student.train() # On remet l'étudiant en mode train
            
        # On extrait automatiquement le nombre de canaux (index 1 de la shape [B, C, H, W])
        feature_dims_student = [f.shape[1] for f in s_feats]
        feature_dims_teacher = [f.shape[1] for f in t_feats]
        
        print(f"\n[Auto-Config] Dimensions extraites de l'étudiant : {feature_dims_student}")
        print(f"[Auto-Config] Dimensions extraites du professeur : {feature_dims_teacher}\n")
        
        # Sécurité : vérifier que les deux modèles renvoient bien le même nombre de tenseurs
        if len(feature_dims_student) != len(feature_dims_teacher):
            raise ValueError(f"Erreur: l'étudiant renvoie {len(feature_dims_student)} features, " 
                             f"mais le prof en renvoie {len(feature_dims_teacher)}.")

        # 3. Création des adaptateurs dynamiquement
        self.adapters = nn.ModuleList([
            nn.Conv2d(s, t, kernel_size=1) 
            for s, t in zip(feature_dims_student, feature_dims_teacher)
        ])

    def train(self, mode=True):
        """Override pour s'assurer que le professeur reste en eval()"""
        super().train(mode)
        self.teacher.eval()

    def forward(self, x):
        s_logits, s_feats = self.student(x)
        with torch.no_grad():
            t_logits, t_feats = self.teacher(x)
        
        return s_logits, t_logits, s_feats, t_feats

def distillation_loss(s_logits, t_logits, s_feats, t_feats, adapters, T=3.0, alpha=0.5, beta=1.0):
    # 1. Standard KL Divergence for Logits
    soft_log_probs = F.log_softmax(s_logits / T, dim=1)
    soft_targets = F.softmax(t_logits / T, dim=1)
    distill_loss = F.kl_div(soft_log_probs, soft_targets, reduction='batchmean') * (T**2)
    
    feat_loss = 0
    for i, adapter in enumerate(adapters):
        projected_s = adapter(s_feats[i])
        feat_loss += F.mse_loss(projected_s, t_feats[i])
        
    return alpha * distill_loss + beta * feat_loss