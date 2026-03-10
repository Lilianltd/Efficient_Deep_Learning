from models import *
import os
from quant.binary_connect_lilian.main2 import main

net_test = [VGG('VGG19'),  ResNet18(),  PreActResNet18(),  GoogLeNet(), densenet_cifar(), DenseNet121(), ResNeXt29_2x64d(), MobileNet()
  ,MobileNetV2()
  ,DPN92()
  ,SENet18()
  ,ShuffleNetV2(1)
  ,EfficientNetB0()
  ,RegNetX_200MF()
  ,SimpleDLA()]

from quant.binary_connect_lilian import binaryconnect

name = [
    "VGG19",
    "ResNet18_quant",
    "PreActResNet18",
    "GoogLeNet",
    'DenseNet_cifar',
    "DenseNet121",
    "ResNeXt29_2x64d",
    "MobileNet",
    "MobileNetV2",
    "DPN92",
    "SENet18",
    "ShuffleNetV2",
    "EfficientNetB0",
    "RegNetX_200MF",
    "SimpleDLA"
]

batch_test = {1 : {'net' : net_test[1], 'name': name[1], "param" : {"alpha":0.4,"lr":0.01, "weight_decay":5e-4, "momentum":0.9}}}

for k, element in batch_test.items():
  para = element["param"]
  net = binaryconnect.BC(element["net"])
  main(net,para, f'{element["name"]}_{para["alpha"]}_{para["lr"]}_{para["weight_decay"]}_{para["momentum"]}')
  