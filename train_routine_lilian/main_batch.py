from models import *
from train_routine_lilian.main import main
from custom_utils import load_model, score
from distillation.distillation_train import main as main_distillation
from distillation.custom_mobilnet_distillation import main as main_distillation_mobilnet

net_test = [MobileNetV2_Custom_Distillation_advanced(width_mult=0.5, depth_mult=1)
  ,MobileNetV2_Custom_Distillation_advanced(width_mult=0.75, depth_mult=0.5)
  ,MobileNetV2_Custom_Distillation_advanced(width_mult=0.5, depth_mult=0.75)
]

name = [
    "MobileNetV2_custom_0.5_1_disti",
    "MobileNetV2_custom_0.75_0.5_disti",
    "MobileNetV2_custom_0.5_0.75_disti"
]

batch_test = {1 : {'net' : net_test[0], 'name': name[0], "param" : {"alpha":0.4,"lr":0.01, "weight_decay":5e-4, "momentum":0.9}, "distillation":{"teacher_path":"./checkpoint/MobileNetV2/ckpt.pth","model_obj":MobileNetV2_Custom_Distillation_advanced, "args":{"width_mult":1, "depth_mult":1}}},
              2 : {'net' : net_test[1], 'name': name[1], "param" : {"alpha":0.4,"lr":0.01, "weight_decay":5e-4, "momentum":0.9}, "distillation":{"teacher_path":"./checkpoint/MobileNetV2/ckpt.pth","model_obj":MobileNetV2_Custom_Distillation_advanced, "args":{"width_mult":1, "depth_mult":1}}},
              3 : {'net' : net_test[2], 'name': name[2], "param" : {"alpha":0.4,"lr":0.01, "weight_decay":5e-4, "momentum":0.9}, "distillation":{"teacher_path":"./checkpoint/MobileNetV2/ckpt.pth","model_obj":MobileNetV2_Custom_Distillation_advanced, "args":{"width_mult":1, "depth_mult":1}}}}

for k, element in batch_test.items():
  para = element["param"]
  if "distillation" in element.keys():
    teacher_model = load_model(element["distillation"]["teacher_path"], element["distillation"]["model_obj"], element["distillation"]["args"])    
    teacher_model = teacher_model.to('cuda')
    if isinstance(teacher_model, MobileNetV2_Custom_Distillation_advanced):
      main_distillation_mobilnet(element["net"], teacher_model, para, f'{element["name"]}_{para["alpha"]}_{para["lr"]}_{para["weight_decay"]}_{para["momentum"]}')
    else:
      main_distillation(element["net"], teacher_model, para, f'{element["name"]}_{para["alpha"]}_{para["lr"]}_{para["weight_decay"]}_{para["momentum"]}')
  else:
    main(element["net"],para, f'{element["name"]}_{para["alpha"]}_{para["lr"]}_{para["weight_decay"]}_{para["momentum"]}')
