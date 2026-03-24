from models import *
from custom_utils import score
import torch

loaded_cpt = torch.load('/homes/l22letar/EDL/pytorch-cifar/checkpoint/MobileNetV2_custom_0.5_0.5_0.4_0.01_0.0005_0.9/ckpt.pth')
new_state_dict = {
    k.replace("module.", ""): v
    for k, v in loaded_cpt["net"].items()
}
model = MobileNetV2_Custom(width_mult=1, depth_mult=1)
model.load_state_dict(new_state_dict)
model = model.to("cuda").half()
model.eval()

from train_routine_lilian.main import load_data, test

_, testloader, _ = load_data()
criterion = torch.nn.CrossEntropyLoss()
import time
s = time.time()
test(model, 0, testloader, "", criterion, 0, True)
print(score(model, 0, 0, 16, 16))
