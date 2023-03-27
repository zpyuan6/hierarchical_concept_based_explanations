import torchvision

DATASET_LIST = {"ADE":"F:\\Automatic_Concept_Explanation\\ADEChallengeData2016\\scene_categories"}

def dataset_load(name):
    if name in DATASET_LIST:
        return torchvision.datasets.ImageFolder()
    else:
        raise(f"Can not find dataset {name}.")