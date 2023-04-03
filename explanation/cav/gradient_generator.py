import os
import torch
import torch.nn as nn
import copy

from explanation.cav.model_hook_manager import ModelHookManager


class GradientGenerator:
    def __init__(self, model:nn.Module, device=None, loss=nn.CrossEntropyLoss(), source_dir:str=None, grad_dir:str=None) -> None:
        self.model = model
        # self.source_dir = source_dir
        # self.grad_dir = grad_dir if grad_dir else os.path.join(self.source_dir, "gradients")
        self.hook_manager = ModelHookManager()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = loss
        model.to(device=self.device)
        model.eval()

    def get_gradient(self, bottleneck_name, example, class_id=None) -> torch.Tensor :
        """Return the gradient of the loss with respect to the bottleneck_name.
        Args:
            y: index of the logit layer
            bottleneck_name: name of the bottleneck to get gradient wrt.
            example: input example. Unused by default. Necessary for getting gradients
                from certain models, such as BERT.
        Returns:
            the gradient array.
        """
        self.hook_manager.register_backward_hook(self.model, bottleneck_name)

        example = example.to(self.device)
        output = self.model(example)
        class_id = torch.argmax(output,dim=1) if class_id == None else torch.tensor([class_id]).to(self.device)

        loss = self.loss(output,class_id)
        loss.backward()

        gradients = copy.deepcopy(self.hook_manager.gradients[0].cpu())

        self.model.zero_grad()
        self.hook_manager.remove_hook()

        return gradients