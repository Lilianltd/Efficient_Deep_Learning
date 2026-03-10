from models import *
from main import main

net_test = [MobileNetV2()
  ,MobileNetV2_Custom(width_mult=0.5, depth_mult=0.5)
  ,MobileNetV2_Custom(width_mult=0.75, depth_mult=0.75)
  ,MobileNetV2_Custom(width_mult=0.75, depth_mult=0.5)
  ,MobileNetV2_Custom(width_mult=0.5, depth_mult=0.75)
]

name = [
    "MobileNetV2",
    "MobileNetV2_custom_0.5_0.5",
    "MobileNetV2_custom_0.75_0.75",
    "MobileNetV2_custom_0.75_0.5",
    "MobileNetV2_custom_0.5_0.75"
]

batch_test = {2 : {'net' : net_test[2], 'name': name[2], "param" : {"alpha":0.4,"lr":0.01, "weight_decay":5e-4, "momentum":0.9}},
              3 : {'net' : net_test[3], 'name': name[3], "param" : {"alpha":0.4,"lr":0.01, "weight_decay":5e-4, "momentum":0.9}},
              4 : {'net' : net_test[4], 'name': name[4], "param" : {"alpha":0.4,"lr":0.01, "weight_decay":5e-4, "momentum":0.9}}}

for k, element in batch_test.items():
  para = element["param"]
  main(element["net"],para, f'{element["name"]}_{para["alpha"]}_{para["lr"]}_{para["weight_decay"]}_{para["momentum"]}')
