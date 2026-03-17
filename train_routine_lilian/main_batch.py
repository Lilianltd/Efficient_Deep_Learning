from models import *
from train_routine_lilian.main import main
from custom_utils import load_model
from distillation.distillation_train import main as main_distillation

net_test = [MobileNetV2_Custom(width_mult=0.5, depth_mult=0.5)
  ,MobileNetV2_Custom(width_mult=0.5, depth_mult=1)
  ,MobileNetV2_Custom(width_mult=0.75, depth_mult=0.5)
  ,MobileNetV2_Custom(width_mult=0.5, depth_mult=0.75)
]

name = [
    "MobileNetV2_custom_0.5_0.5",
    "MobileNetV2_custom_0.5_1",
    "MobileNetV2_custom_0.75_0.5",
    "MobileNetV2_custom_0.5_0.75"
]

batch_test = {2 : {'net' : net_test[0], 'name': name[0], "param" : {"alpha":0.4,"lr":0.01, "weight_decay":5e-4, "momentum":0.9}, "distillation":{"teacher_path":"./saved_checkpoint/ResNet18_0.4_0.01_0.0005_0.9/ckpt.pth","model_obj":ResNet18}}}
for k, element in batch_test.items():
  para = element["param"]
  if "distillation" in element.keys():
    teacher_model = load_model(element["distillation"]["teacher_path"], element["distillation"]["model_obj"], {})
    main_distillation(element["net"], teacher_model, para, f'{element["name"]}_{para["alpha"]}_{para["lr"]}_{para["weight_decay"]}_{para["momentum"]}')
  else:
    main(element["net"],para, f'{element["name"]}_{para["alpha"]}_{para["lr"]}_{para["weight_decay"]}_{para["momentum"]}')
