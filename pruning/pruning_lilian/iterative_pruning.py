from models import MobileNetV2_Custom
import torch
import torch.nn.utils.prune as prune
from custom_utils import score
from train_routine_lilian.main import main

RESNET_PATH = '/homes/l22letar/EDL/pytorch-cifar/checkpoint/MobileNetV2_custom_0.4_0.01_0.0005_0.9/ckpt.pth'

def load_resnet_and_make_permanent(path):
    checkpoint = torch.load(path)
    checkpoint = {
        k.replace("module.", ""): v
        for k, v in checkpoint["net"].items()
    }
    # If saved as a full dict, get the state_dict part
    state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint
    
    new_state_dict = {}
    for key in list(state_dict.keys()):
        if key.endswith('_orig'):
            # Generate the standard weight by multiplying orig * mask
            base_name = key[:-5] # remove '_orig'
            mask_key = base_name + '_mask'
            new_state_dict[base_name] = state_dict[key] * state_dict[mask_key]
        elif key.endswith('_mask'):
            continue # skip the mask, we've already used it
        else:
            new_state_dict[key] = state_dict[key]
            
    model = MobileNetV2_Custom(width_mult=0.5)
    model.load_state_dict(new_state_dict)
    return model

def load_resnet(path):
    loaded_cpt = torch.load(path)
    model = MobileNetV2_Custom(width_mult=0.5)
    new_state_dict = {
        k.replace("module.", ""): v
        for k, v in loaded_cpt["net"].items()
    }
    model.load_state_dict(new_state_dict)
    model.to("cuda")
    return model

def resnet_pruning(model, s_rate=0.2, u_rate=0.3):
    # 1. Local Structured Pruning (Fixed % per layer)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Note: For ResNet, pruning 'dim=1' (input channels) is 
            # often safer for the residual connections than 'dim=0'.
            prune.ln_structured(module, name="weight", amount=s_rate, n=2, dim=0)

    # 2. Global Unstructured Pruning (The "Fine-tuning" phase)
    params_to_prune = [
        (m, "weight") for m in model.modules() if isinstance(m, torch.nn.Conv2d)
    ]
    
    prune.global_unstructured(
        params_to_prune, 
        pruning_method=prune.L1Unstructured, 
        amount=u_rate
    )

def resnet_restore(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            prune.remove(layer, "weight")
        

if __name__ == "__main__":
    from main import load_data, test

    _, testloader, _ = load_data()
    criterion = torch.nn.CrossEntropyLoss()
    import time
    s = time.time()
    import numpy as np

    # --- Configuration ---
    target_ratios_str = [0.15]
    target_ratios_un = [0.6] 
    steps = 5

    for target_s in target_ratios_str:
        for target_u in target_ratios_un:
            model = load_resnet(RESNET_PATH)
            for s in range(steps-1, 0, -1):
                coeff = -(s-1)**2/(steps-1)**2 + 1
                
                # Pruning partiel
                element = {'net' : model, 'name': f"Mobilpruning_un_{target_u*coeff}_str_{target_s*coeff}", "param" : {"alpha":0.4,"lr":0.0001, "weight_decay":5e-4, "momentum":0.9}}
                para = element["param"]
                model.half()
                resnet_pruning(model, s_rate=target_s*coeff, u_rate=target_u*coeff)
                _,_,acc_prune,_ = test(model, 0, testloader, "", criterion, 0, half=True)
                model.float()
                element = {'net' : model, 'name': f"Mobilpruning_un_{target_u*coeff}_str_{target_s*coeff}", "param" : {"alpha":0.4,"lr":0.0001, "weight_decay":5e-4, "momentum":0.9}}
                main(element["net"],para, f'{element["name"]}',10)
                
                model = load_resnet_and_make_permanent(f'./checkpoint/Mobilpruning_un_{target_u*coeff}_str_{target_s*coeff}/ckpt.pth')
                model.to("cuda").eval().half()
                score_value = score(model, target_s*coeff,target_u*coeff*(1-target_s*coeff), 16, 16)
                _,_,acc_fine_tuned,_ = test(model, 0, testloader, "", criterion, 0,half=True)
                a = [target_s*coeff, target_u*coeff, acc_prune, acc_fine_tuned, score_value]
                if acc_fine_tuned < 88:
                    break
                import csv                
                with open("output_pruning.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(a)
