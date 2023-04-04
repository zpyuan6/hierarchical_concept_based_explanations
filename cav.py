import argparse
import os
import numpy as np
from torchvision.datasets import ImageNet

from models.model_load import model_load
from utils.utils import make_dir_if_not_exists, load_yaml
from explanation.cav.activation_generator import ActivationGenerator
from explanation.cav.cav import get_or_train_cav, CAV
from explanation.cav.tcav import compute_tcav_score, get_directional_dir

# Reference docs https://github.com/tensorflow/tcav/blob/b922c44bcc64c6bdddb8f661d732fa2145c99d95/Run_TCAV.ipynb
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create examples and concepts folders.')
    parser.add_argument('--source_dir', type=str, default="F:\\hierarchical_concept_explanation\\",
                        help='Name for the directory where we will create the data.')
    parser.add_argument('--number_of_images_per_folder', type=int, default=50,
                        help='Number of images to be included in each folder')
    parser.add_argument('--number_of_random_folders', type=int, default=3,
                        help='Number of folders with random examples that we will generate for tcav')
    parser.add_argument('--yaml', type=str, help='Load parameters from yaml')

    args = parser.parse_args()
    if args.yaml != None:
        yaml = load_yaml(args.yaml)
        parser.set_defaults(**yaml)
        args = parser.parse_args()

    print("Args is: ", args.__dict__)
    # source_dir: where images of concepts, target class and random images (negative samples when learning CAVs) live. Each should be a sub-folder within this directory. 
    source_dir = args.source_dir
    
    # directory to store CAVs
    cav_dir = os.path.join(source_dir, "cavs")
    make_dir_if_not_exists(cav_dir)

    activation_dir = os.path.join(source_dir, "activations")
    make_dir_if_not_exists(activation_dir)
    bottlenecks = ['Mixed_5d', 'Conv2d_2a_3x3']

    # this is a regularizer penalty parameter for linear classifier to get CAVs. 
    alphas = [0.1]

    target = 'zebra'
    concepts = ["dotted","striped","zigzagged"]  
    # random_counterpart = 'random500_1'
    model_name = args.model_name
    # LABEL_PATH = './imagenet_comp_graph_label_strings.txt'
    imagenet_dataset_path = "F:\\ImageNet"

    model = model_load(model_name)

    # Get acts
    activation_generator = ActivationGenerator(model,source_dir)
    acts = activation_generator.process_and_load_activations(bottlenecks, concepts+[target])

    # Get CAVs
    cav_instance = get_or_train_cav(
                    concepts,
                    bottlenecks[0],
                    acts,
                    cav_dir=cav_dir,
                    cav_hparams=None,
                    overwrite=False)

    # clean up
    for c in concepts:
        del acts[c]

    # Hypo testing
    cav_hparams = CAV.default_hparams()
    a_cav_key = CAV.cav_key(concepts, bottlenecks[0], cav_hparams['model_type'], cav_hparams['alpha'])

    target_class_for_compute_tcav_score = target

    cav_concept = concepts[0]

    class_dir = ImageNet(imagenet_dataset_path, download=False).class_to_idx
    id_to_class_dir = {value:key for (key,value) in class_dir.items()}
    class_id = class_dir[target_class_for_compute_tcav_score]

    i_up = compute_tcav_score(
        model, 
        class_id,
        cav_concept,
        cav_instance, 
        activation_generator.get_examples_for_concept(target),
        )

    print("i_up", i_up)

    val_directional_dirs = get_directional_dir(
        model,
        class_id,
        cav_concept,
        cav_instance,
        activation_generator.get_examples_for_concept(target)
        )

    print("val_directional_dirs", val_directional_dirs)

    result = {
        'cav_key':
            a_cav_key,
        'cav_concept':
            cav_concept,
        'negative_concept':
            concepts[1],
        'target_class':
            bottlenecks[0],
        'cav_accuracies':
            cav_instance.accuracies,
        'i_up':
            i_up,
        'val_directional_dirs_abs_mean':
            np.mean(np.abs(val_directional_dirs)),
        'val_directional_dirs_mean':
            np.mean(val_directional_dirs),
        'val_directional_dirs_std':
            np.std(val_directional_dirs),
        'val_directional_dirs':
            val_directional_dirs,
        'bottleneck':
            bottlenecks[0]
    }

    del acts

    # return result