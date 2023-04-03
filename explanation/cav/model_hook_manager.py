import torch.nn as nn

class ModelHookManager():
    def __init__(self) -> None:
        self.features = None
        self.hooks = []
        self.gradients = None

    def hook(self, module, input, output):
        if self.features != None:
            self.features.append(output.clone())
        else:
            self.features = [output.clone()]

    def register_hook(self,model:nn.Module,layer_name:str):
        hook = model._modules[layer_name].register_forward_hook(self.hook)
        self.hooks.append(hook)

    def backward_hook(self, module, grad_in, grad_out):
        if self.gradients != None:
            self.gradients.append(grad_out[0].clone())
        else:
            self.gradients = [grad_out[0].clone()]

    def register_backward_hook(self,model:nn.Module,layer_name:str):
        hook = model._modules[layer_name].register_backward_hook(self.backward_hook)
        self.hooks.append(hook)

    def remove_hook(self):
        for hook in self.hooks:
            hook.remove()
        self.features = None
        self.gradients = None

    def clean_features(self):
        del self.features
        del self.gradients
        self.features = None
        self.gradients = None