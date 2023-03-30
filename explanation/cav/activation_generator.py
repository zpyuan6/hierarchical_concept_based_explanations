import os
from tqdm import tqdm
from PIL import Image 
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ModelHookManager():
    def __init__(self) -> None:
        self.features = None
        self.hooks = []

    def hook(self, module, input, output):
        if self.features != None:
            self.features.append(output.clone())
        else:
            self.features = [output.clone()]

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
    def __init__(self, model:nn.Module, acts_dir:str, input_size=[299,299], device=None) -> None:
        self.model = model
        self.acts_dir = acts_dir
        self.hook_manager = ModelHookManager()
        self.interested_layer_names = interested_layer_names
        for interested_layer_name in self.interested_layer_names:
            self.hook_manager.register_hook(self.model, interested_layer_name)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size

    def get_activations_for_folder(self, dataset_path, concepts:list, interested_layer_names:list, is_save=True):
        """
        return 
            activation_results: activation objects {""}
        """
        acts = {}

        for concept in concepts:
            if concept not in acts:
                acts[concept] = {}
            for bottleneck_name in self.interested_layer_names:
                acts_path = os.path.join(self.acts_dir, 'acts_{}_{}'.format(concept, bottleneck_name)) if self.acts_dir else None
                if acts_path and os.path.exists(acts_path):
                    # Load activations from acts_path for a certain concept and a bottleneck
                    with open(acts_path, 'rb') as f:
                        acts[concept][bottleneck_name] = np.load(f, allow_pickle=True).squeeze()
                        print(('Loaded {} shape {}'.format(acts_path, acts[concept][bottleneck_name].shape)))
                else:
                    acts[concept][bottleneck_name] = self.get_activations_for_concept(concept, bottleneck_name)

        return acts

    def get_activations_for_concept(self, concept, bottleneck):
        # activation_save_paths = []
        # for interested_layer_name in self.interested_layer_names:
        #     activation_save_path = os.path.join(self.acts_dir,interested_layer_name,dataset_path.split("\\")[-1])
        #     activation_save_paths.append(activation_save_path)
        #     print("activation_save_path: ", activation_save_path)
        #     if is_save and (not os.path.exists(activation_save_path)):
        #         os.makedirs(activation_save_path)

        # crop = transforms.Compose([
        #     transforms.Resize(self.input_size),
        #     transforms.ToTensor()
        # ])

        # self.model.to(self.device)
        # self.model.eval()

        # activation_results = {}
        # for root, folders, files in os.walk(dataset_path):
        #     with tqdm(total=len(files),desc=f"Start processing images under {root}") as tbar:
        #         for file in files:
        #             with torch.no_grad():
        #                 img = Image.open(os.path.join(root,file))
        #                 img = img.convert('RGB')
        #                 img = crop(img).unsqueeze(0).to(self.device)
        #                 prediction = self.model(img)
        #                 tbar.update(1)
                        
        #                 for i,activation_save_path in enumerate(activation_save_paths):
        #                     if is_save:
        #                         torch.save(self.hook_manager.features[i].cpu(),os.path.join(activation_save_path,f"{file.split('.')[0]}.pt"))
                            
        #                     if not interested_layer_name[i] in activation_results:
        #                         activation_results[interested_layer_name[i]] = {dataset_path.split("\\")[-1]:[self.hook_manager.features[i].cpu()]}
        #                     else:
        #                         activation_results[interested_layer_name[i]][dataset_path.split("\\")[-1]].append(self.hook_manager.features[i].cpu())
        #                 self.hook_manager.clean_features()
                        
        return acts