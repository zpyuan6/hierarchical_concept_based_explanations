import os
import torch
import torch.nn as nn

from model_hook_manager import ModelHookManager


class GradientGenerator:
    def __init__(self, model:nn.Module, source_dir:str, acts_dir:str=None, input_size=[299,299], device=None) -> None:
        self.model = model
        self.source_dir = source_dir
        self.acts_dir = acts_dir if acts_dir else os.path.join(self.source_dir, "activations")
        self.hook_manager = ModelHookManager()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size


    def get_gradient(self, act, [class_id], cav.bottleneck, example):
        """Return the gradient of the loss with respect to the bottleneck_name.
        Args:
            acts: activation of the bottleneck
            y: index of the logit layer
            bottleneck_name: name of the bottleneck to get gradient wrt.
            example: input example. Unused by default. Necessary for getting gradients
                from certain models, such as BERT.
        Returns:
            the gradient array.
        """