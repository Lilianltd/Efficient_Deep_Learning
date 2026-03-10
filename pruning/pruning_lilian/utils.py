import torch
import torch.nn.utils.prune as prune

def load_and_make_permanent(path, ModelClass, args_model):
    checkpoint = torch.load(path)
    checkpoint = {
        k.replace("module.", ""): v
        for k, v in checkpoint["net"].items()
    }
    # If saved as a full dict, get the state_dict part
    state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint
    
    new_state_dict = {}
    for key in list(state_dict.keys()):
        if key.endswith('_orig'):
            # Generate the standard weight by multiplying orig * mask
            base_name = key[:-5] # remove '_orig'
            mask_key = base_name + '_mask'
            new_state_dict[base_name] = state_dict[key] * state_dict[mask_key]
        elif key.endswith('_mask'):
            continue # skip the mask, we've already used it
        else:
            new_state_dict[key] = state_dict[key]
            
    model = ModelClass(**args_model)
    model.load_state_dict(new_state_dict)
    return model

def pruning(model, s_rate=0.2, u_rate=0.3):
    # 1. Local Structured Pruning (Fixed % per layer)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Note: For ResNet, pruning 'dim=1' (input channels) is 
            # often safer for the residual connections than 'dim=0'.
            prune.ln_structured(module, name="weight", amount=s_rate, n=2, dim=0)

    # 2. Global Unstructured Pruning (The "Fine-tuning" phase)
    params_to_prune = [
        (m, "weight") for m in model.modules() if isinstance(m, torch.nn.Conv2d)
    ]
    
    prune.global_unstructured(
        params_to_prune, 
        pruning_method=prune.L1Unstructured, 
        amount=u_rate
    )
    
def restore(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            prune.remove(layer, "weight")
        
