from models import ResNet18, DenseNet121, MobileNetV2, MobileNet, MobileNetV2_Custom
from score import score
import torch

#loaded_cpt = torch.load('/homes/l22letar/EDL/pytorch-cifar/checkpoint/ResNet18_0.4_0.01_0.0005_0.9/ckpt.pth')
#new_state_dict = {
#    k.replace("module.", ""): v
#    for k, v in loaded_cpt["net"].items()
#}
model = ResNet18()
#model.load_state_dict(new_state_dict)
model = model.to("cuda").half()
model.eval()

from main import load_data, test

_, testloader, _ = load_data()
criterion = torch.nn.CrossEntropyLoss()
import time
s = time.time()
#test(model, 0, testloader, "", criterion, 0, False)
print(time.time()-s)
print(score(model, 0.7, 0.3*0.7, 16, 16))
