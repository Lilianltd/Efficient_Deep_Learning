import torch
import time
from models import *
from custom_utils import score, load_model
from train_routine_lilian.main import main, test
from train_routine_lilian.utils import load_data
from .utils import *

MODEL_PATH = '/homes/l22letar/EDL/pytorch-cifar/saved_checkpoint/MobileNetV2_custom_0.4_0.01_0.0005_0.9/ckpt.pth'
ModelClass = MobileNetV2_Custom
model_args = {"width_mult":0.5}
model_name = "Mobilnet"
        
if __name__ == "__main__":
    _, testloader, _ = load_data()
    criterion = torch.nn.CrossEntropyLoss()

    s = time.time()

    ratios_unstructured = [0.5]
    ratios_str = [0.1]
    result = []
    for ratio_value_str in ratios_str:
        if ratio_value_str == 0:
            continue
        for ratio_value_unstructured in ratios_unstructured:
            model = load_model(MODEL_PATH, ModelClass, model_args)
            element = {'net' : model, 'name': f"{model_name}_Pruning_un_{ratio_value_unstructured}_str_{ratio_value_str}", "param" : {"alpha":0.4,"lr":0.0001, "weight_decay":5e-4, "momentum":0.9}}
            para = element["param"]

            model.eval().half()
            
            pruning(model, ratio_value_str, ratio_value_unstructured)
            restore(model)
            _,_,acc_prune,_ = test(model, 0, testloader, "", criterion, 0, True)
            score_value = score(model, ratio_value_str,ratio_value_unstructured*(1-ratio_value_str), 16, 16)
            model = load_model(MODEL_PATH, ModelClass, model_args)
            pruning(model, ratio_value_str, ratio_value_unstructured)
            element = {'net' : model, 'name': f"{model_name}_Pruning_un_{ratio_value_unstructured}_str_{ratio_value_str}", "param" : {"alpha":0.4,"lr":0.0001, "weight_decay":5e-4, "momentum":0.9}}
            main(element["net"],para, f'{element["name"]}_{para["alpha"]}_{para["lr"]}_{para["weight_decay"]}_{para["momentum"]}',20)
            
            model = load_and_make_permanent(f'./checkpoint/{element["name"]}_{para["alpha"]}_{para["lr"]}_{para["weight_decay"]}_{para["momentum"]}/ckpt.pth', ModelClass, model_args)
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
