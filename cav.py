import argparse
import os
import pickle
import yaml

from models.model_load import model_load
from utils.utils import make_dir_if_not_exists, load_yaml
from explanation.cav.activation_generator import ActivationGenerator
# from explanation.cav.cav import get_or_train_cav, CAV
from explanation.cav.tcav import TCAV

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
        yaml_file = load_yaml(args.yaml)
        parser.set_defaults(**yaml_file)
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

    activation_generator = ActivationGenerator(model,source_dir)

    tcav_generator = TCAV(target,
                        concepts, 
                        bottlenecks,
                        activation_generator,
                        cav_dir,
                        num_random_exp=3
                        )

    results = tcav_generator.run(run_parallel=False)

    print(results)

    with open("results.yaml",'w') as results_file:
        yaml.dump(results, results_file, default_flow_style=False)