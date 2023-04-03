import os,sys
from tqdm import tqdm
from PIL import Image 
import numpy as np
from multiprocessing import dummy as multiprocessing

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from explanation.cav.model_hook_manager import ModelHookManager

class ActivationGenerator:
    """ Downloads an image.
    Downloads and image from a image url provided and saves it under path.
    Filters away images that are corrupted or smaller than 10KB
    Args:
        model: Path to the folder where we're saving this image.
        acts_dir: path for saving activations
        interested_layer_name: 
    """
    def __init__(self, model:nn.Module, source_dir:str, acts_dir:str=None, input_size=[299,299], device=None) -> None:
        self.model = model
        self.source_dir = source_dir
        self.acts_dir = acts_dir if acts_dir else os.path.join(self.source_dir, "activations")
        self.hook_manager = ModelHookManager()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size

    def process_and_load_activations(self, interested_layer_names:list, concepts:list):
        """
        return 
            activation_results: activation objects {""}
        """
        acts = {}

        for concept in concepts:
            if concept not in acts:
                acts[concept] = {}
            for bottleneck_name in interested_layer_names:
                acts_path = os.path.join(self.acts_dir, 'acts_{}_{}'.format(concept, bottleneck_name))
                if os.path.exists(acts_path):
                    # Load activations from acts_path for a certain concept and a bottleneck
                    with open(acts_path, 'rb') as f:
                        acts[concept][bottleneck_name] = np.load(f, allow_pickle=True).squeeze()
                        print(('Loaded {} shape {}'.format(acts_path, acts[concept][bottleneck_name].shape)))
                else:
                    # Process activations
                    acts[concept][bottleneck_name] = self.get_activations_for_concept(concept, bottleneck_name)
                    print('{} does not exist, Making one...'.format(acts_path))
                    with open(acts_path, 'wb') as f:
                        np.save(f, acts[concept][bottleneck_name])

        return acts

    def get_examples_for_concept(self, concept):
        concept_dir = os.path.join(self.source_dir,concept)
        img_paths = []

        for root,folders,files in os.walk(concept_dir):
            for file in files:
                img_paths.append(os.path.join(concept_dir,file))
        
        imgs = self.load_image_from_files(img_paths)

        return imgs

    def load_image_from_file(self, file_path):
        crop = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])

        img = Image.open(file_path)
        img = img.convert('RGB')
        img = crop(img).unsqueeze(0)

        return img

    def load_image_from_files(self, files_list, do_shuffle=True, run_parallel=True):
        imgs=[]
        filesnames = files_list[:]

        if do_shuffle:
            np.random.shuffle(filesnames)

        if run_parallel:
            pool = multiprocessing.Pool()
            imgs = pool.map(
                lambda filename: self.load_image_from_file(filename),
                filesnames[:])
            pool.close()
            imgs = [img for img in imgs if img is not None]
            if len(imgs) <= 1:
                raise ValueError(
                    'You must have more than 1 image in each class to run TCAV.')
        else:
            for filename in files_list:
                img = self.load_image_from_file(filename)
            if img is not None:
                imgs.append(img)

            if len(imgs) <= 1:
                raise ValueError(
                    'You must have more than 1 image in each class to run TCAV.')

        return imgs

    def get_activations_for_concept(self, concept, bottleneck):
        examples = self.get_examples_for_concept(concept)

        self.hook_manager.register_hook(self.model, bottleneck)

        self.model.to(self.device)
        self.model.eval()

        activation_results = []
        with tqdm(total=len(examples)) as tbar:
            for example in examples:
                with torch.no_grad():
                    input = example.to(self.device)
                    prediction = self.model(input)
                    tbar.update(1)
                    activation_results.append(self.hook_manager.features[0].cpu().numpy())
                    self.hook_manager.clean_features()

        self.hook_manager.remove_hook()

        return np.array(activation_results)