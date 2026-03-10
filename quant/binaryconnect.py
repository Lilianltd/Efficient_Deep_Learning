import torch
import torch.nn as nn

class BC():
    def __init__(self, model):
        self.model = model 
        self.saved_params = [] 
        self.target_modules = []

        # We can build the lists in a single, clean pass
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Clone the data to initialize the saved real-valued weights
                self.saved_params.append(m.weight.data.clone())
                # Save a reference to the parameter itself
                self.target_modules.append(m.weight)
                
        self.num_of_params = len(self.target_modules)

    def save_params(self):
        """Saves current real-valued weights into saved_params"""
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarization(self):
        """Saves real weights, then binarizes the model's weights"""
        self.save_params()
        for index in range(self.num_of_params):
            w = self.target_modules[index].data
            # sign() leaves 0s as 0. The add_ turns exactly 0 into 1.
            w.copy_(w.sign().add_(w.eq(0).float()))

    def restore(self):
        """Restores the real-valued weights back into the model"""
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
    
    def clip(self):
        """Clips the real-valued weights to the [-1, 1] range in-place"""
        for index in range(self.num_of_params):
            self.target_modules[index].data.clamp_(-1.0, 1.0)

    def forward(self, x):
        """Wrapper for the forward pass"""
        return self.model(x)