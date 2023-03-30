import torch
import torchvision


def model_load(name):
    if name == "inceptionv3":
        model = torchvision.models.inception_v3(pretrained=True)
    
    return model