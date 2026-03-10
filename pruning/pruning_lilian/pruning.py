from models import *
import torch
import torch.nn.utils.prune as prune
from custom_utils import score
from train_routine_lilian.main import main, test
from train_routine_lilian.utils import load_data
from custom_utils import load_model
import time

RESNET_PATH = '/homes/l22letar/EDL/pytorch-cifar/saved_checkpoint/MobileNetV2_custom_0.4_0.01_0.0005_0.9/ckpt.pth'
model_type = MobileNetV2_Custom
model_kwargs = {"width_mult":0.5}

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
    _, testloader, _ = load_data()
    criterion = torch.nn.CrossEntropyLoss()

    s = time.time()

    ratios_unstructured = [0.5]
    ratios_str = [0.1]
    print(ratios_unstructured, ratios_str)

    result = []

    for ratio_value_str in ratios_str:
        if ratio_value_str == 0:
            continue
        for ratio_value_unstructured in ratios_unstructured:
            model = load_model(RESNET_PATH, model_type, model_kwargs)
            element = {'net' : model, 'name': f"ResNetPruning_un_{ratio_value_unstructured}_str_{ratio_value_str}", "param" : {"alpha":0.4,"lr":0.0001, "weight_decay":5e-4, "momentum":0.9}}
            para = element["param"]

            model.eval().half()
            
            resnet_pruning(model, ratio_value_str, ratio_value_unstructured)
            resnet_restore(model)
            _,_,acc_prune,_ = test(model, 0, testloader, "", criterion, 0, True)
            score_value = score(model, ratio_value_str,ratio_value_unstructured*(1-ratio_value_str), 16, 16)
            model = load_model(RESNET_PATH, model_type, model_kwargs)
            resnet_pruning(model, ratio_value_str, ratio_value_unstructured)
            element = {'net' : model, 'name': f"ResNetPruning_un_{ratio_value_unstructured}_str_{ratio_value_str}", "param" : {"alpha":0.4,"lr":0.0001, "weight_decay":5e-4, "momentum":0.9}}
            main(element["net"],para, f'{element["name"]}_{para["alpha"]}_{para["lr"]}_{para["weight_decay"]}_{para["momentum"]}',20)
            
            model = load_resnet_and_make_permanent(f'./checkpoint/{element["name"]}_{para["alpha"]}_{para["lr"]}_{para["weight_decay"]}_{para["momentum"]}/ckpt.pth')
            model.to("cuda").half()
            _,_,acc_fine_tuned,_ = test(model, 0, testloader, "", criterion, 0, True)
            a = [ratio_value_unstructured, ratio_value_str, acc_prune, acc_fine_tuned, score_value]
            if acc_fine_tuned < 88:
                break
            import csv

            with open("output_pruning.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(a)
                
    print(time.time()-s)
