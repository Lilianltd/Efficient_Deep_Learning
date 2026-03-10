from models import *
import torch
from custom_utils import score, load_model
from train_routine_lilian.main import main, test
from train_routine_lilian.utils import load_data
from .utils import *

#CONFIG

MODEL_PATH = '/homes/l22letar/EDL/pytorch-cifar/saved_checkpoint/MobileNetV2_custom_0.4_0.01_0.0005_0.9/ckpt.pth'
ModelClass = MobileNetV2_Custom
model_args = {"width_mult":0.5}
model_name = "Mobilnet"

if __name__ == "__main__":
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
            model = load_model(MODEL_PATH, ModelClass, model_args)
            for s in range(steps-1, 0, -1):
                coeff = -(s-1)**2/(steps-1)**2 + 1
                
                # Pruning partiel
                element = {'net' : model, 'name': f"{model_name}_pruning_un_{target_u*coeff}_str_{target_s*coeff}", "param" : {"alpha":0.4,"lr":0.0001, "weight_decay":5e-4, "momentum":0.9}}
                para = element["param"]
                model.half()
                pruning(model, s_rate=target_s*coeff, u_rate=target_u*coeff)
                _,_,acc_prune,_ = test(model, 0, testloader, "", criterion, 0, half=True)
                model.float()
                element = {'net' : model, 'name': f"{model_name}_pruning_un_{target_u*coeff}_str_{target_s*coeff}", "param" : {"alpha":0.4,"lr":0.0001, "weight_decay":5e-4, "momentum":0.9}}
                main(element["net"],para, f'{element["name"]}',10)
                
                model = load_and_make_permanent(f'./checkpoint/{model_name}_pruning_un_{target_u*coeff}_str_{target_s*coeff}/ckpt.pth')
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
            restore(model)

