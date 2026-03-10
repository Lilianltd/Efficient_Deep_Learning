import torchinfo
import torch

def score(model, ps, pu, qw, qa):
    model_ori_size = 11173962/2
    model_ori_compute_ops = 555432330/2
    input_data = torch.randn(1, 3, 32, 32).to("cuda").to(torch.float16)
    stats = torchinfo.summary(model, input_data=input_data, verbose=0)
    model_weight = stats.trainable_params
    print(f"model_weight: {model_weight}")
    model_compute_ops = stats.total_mult_adds
    print(f"model_compute_ops: {model_compute_ops}")
    return ((1-ps-pu)*qw/32*model_weight)/model_ori_size + ((1-ps)*max(qw,qa)/32*model_compute_ops)/model_ori_compute_ops


from models import DenseNet121, EfficientNetB0
# checkpoint = torch.load('./checkpoint/ckpt_densenet_lilian_pruned_quantized.pth', map_location='cpu')
checkpoint = torch.load('./checkpoint/ckpt_efficientnetb0_structured_pruned_full.pth', map_location='cpu')
if isinstance(checkpoint, dict):
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
    model = EfficientNetB0()
    model.load_state_dict(state_dict)
else:
    # checkpoint is the model itself
    model = checkpoint
model = model.to('cuda').to(torch.float16)
ps = 0.5
pu = (1-ps)*0.7
# pu = 0.7
qw = 16
qa = 16

print(score(model, ps, pu, qw, qa))