import os
from PIL import Image 
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ModelHookManager():
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
    def __init__(self, model:nn.Module, acts_dir:str, interested_layer_name:str, input_size=[299,299], device=None) -> None:
        self.model = model
        self.acts_dir = acts_dir
        self.hook_manager = ModelHookManager()
        self.interested_layer_name = interested_layer_name
        self.hook_manager.register_hook(self.model, interested_layer_name)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size

    def update_acts_dir(self, acts_dir):
        self.acts_dir = acts_dir

    def get_activations_for_folder(self, dataset_path, is_save=True):
        crop = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])

        for root, folders, files in os.walk(dataset_path):
            for file in files:
                with torch.no_grad():
                    img = Image.open(os.path.join(root,file))
                    img = crop(img).unsqueeze(0).to(self.device)
                    prediction = self.model(img)
                    print(file)

                    torch.save(self.hook_manager.features.cpu(),os.path.join(self.acts_dir,f"{self.interested_layer_name}_{file.split('.')[0]}.pt"))