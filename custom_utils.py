import torch
import torchinfo

def load_model(path, ModelClass, model_kwargs):
    loaded_cpt = torch.load(path)
    model = ModelClass(**model_kwargs)
    
    new_state_dict = {
        k.replace("module.", ""): v
        for k, v in loaded_cpt["net"].items()
    }
    
    model.load_state_dict(new_state_dict)
    model.to("cuda")
    
    return model

def score(model, ps, pu, qw, qa):
    model_ori_size = 11173962/2
    model_ori_compute_ops = 555432330/2
    input_data = torch.randn(1, 3, 32, 32).to("cuda").to(torch.float16)
    stats = torchinfo.summary(model, input_data=input_data, verbose=0)
    model_weight = stats.trainable_params
    model_compute_ops = stats.total_mult_adds
    return ((1-ps-pu)*qw/32*model_weight)/model_ori_size + ((1-ps)*max(qw,qa)/32*model_compute_ops)/model_ori_compute_ops



import os

def load_checkpoint_meta(path, device):
    """Loads a checkpoint and extracts metadata. Works with dicts or naked models."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        # It's an organized checkpoint
        net = checkpoint['net']
        history = checkpoint.get('history', [])
    else:
        # It's a raw model or state dict
        net = checkpoint
        history = []
    
    return net, history, checkpoint

def save_checkpoint_meta(model, history, acc, save_dir, base_name="EfficientNet"):
    """
    Saves a checkpoint with an ordered history of operations.
    history is a list of strings: ['unP50', 'sP30', 'F20', 'Q8']
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct filename based on history
    history_str = "".join(history)
    filename = f"{base_name}_{history_str}.pth" if history_str else f"{base_name}.pth"
    out_path = os.path.join(save_dir, filename)
    
    state_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    # Pack everything
    checkpoint = {
        'net': state_to_save,
        'acc': acc,
        'history': history
    }
    
    torch.save(checkpoint, out_path)
    print(f"==> Checkpoint saved to {out_path}")
    return out_path

import os

def load_checkpoint_meta(path, device='cpu'):
    """Loads a checkpoint and extracts metadata. Works with dicts or naked models."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        # It's an organized checkpoint
        net_dict = checkpoint['net']
        history = checkpoint.get('history', [])
    else:
        # It's a raw model or state dict
        net_dict = checkpoint
        history = []
    
    return net_dict, history, checkpoint

def save_checkpoint_meta(model, history, acc, save_dir, base_name="EfficientNet", **kwargs):
    """
    Saves a checkpoint with an ordered history of operations.
    history is a list of strings: ['unP50', 'sP30', 'F20', 'Q8']
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct filename based on history
    history_str = "".join(history)
    filename = f"{base_name}_{history_str}.pth" if history_str else f"{base_name}.pth"
    out_path = os.path.join(save_dir, filename)
    
    state_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    # Pack everything
    checkpoint = {
        'net': state_to_save,
        'acc': acc,
        'history': history,
        **kwargs
    }
    
    torch.save(checkpoint, out_path)
    print(f"==> Checkpoint saved to {out_path}")
    return out_path
