import torch
import torchinfo

def load_model(path : str, ModelClass, model_kwargs):
    loaded_cpt = torch.load(path)
    model = ModelClass(**model_kwargs)
    
    new_state_dict = {
        k.replace("module.", ""): v
        for k, v in loaded_cpt["net"].items()
    }
    
    model.load_state_dict(new_state_dict)
    model.to("cuda")
    
    return model

def score(model, ps : float, pu : float, qw: float, qa: float) -> float:
    model_ori_size = 11173962/2
    model_ori_compute_ops = 555432330/2
    input_data = torch.randn(1, 3, 32, 32).to("cuda").to(torch.float16)
    stats = torchinfo.summary(model, input_data=input_data, verbose=0)
    model_weight = stats.trainable_params
    model_compute_ops = stats.total_mult_adds
    return ((1-ps-pu)*qw/32*model_weight)/model_ori_size + ((1-ps)*max(qw,qa)/32*model_compute_ops)/model_ori_compute_ops


