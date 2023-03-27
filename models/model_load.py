import torch
import torchvision


def model_load():
    model = torchvision.models.inception_v3(pretrained=True)
    return model