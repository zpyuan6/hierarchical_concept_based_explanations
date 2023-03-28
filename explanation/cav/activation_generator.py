import torch
import torch.nn as nn

class ModelHook():
    def __init__(self) -> None:
        self.features = None
        self.hooks = []

    def hook(self, module, input, output):
        if self.features != None:
            self.features = torch.cat((self.features,output.clone().squeeze().unsqueeze(0)),0)
        else:
            self.features = output.clone().squeeze().unsqueeze(0)

    def register_hook(self,model:nn.Module,layer_name:str):
        hook = model._modules[layer_name].register_forward_hook(self.hook)
        self.hooks.append(hook)

    def remove_hook(self):
        for hook in self.hooks:
            hook.remove()
        self.features = []

    def clean_features(self):
        del self.features
        self.features = None

class ActivationGenerator:
    """ Downloads an image.
    Downloads and image from a image url provided and saves it under path.
    Filters away images that are corrupted or smaller than 10KB
    Args:
        model: Path to the folder where we're saving this image.
        acts_dir: path for saving activations
        interested_layer_name: 
    """
    def __init__(self, model, acts_dir, interested_layer_name) -> None:
        self.model = model
        self.acts_dir = acts_dir

    def get_activations_for_folder(self, is_save=True):
        self.model

    def get_activation_for_example(self, example):
        self.model
        return reshape_activations(acts).squeeze()