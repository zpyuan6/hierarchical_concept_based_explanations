import os
import argparse
import pandas as pd
import urllib.request
import shutil
import random
from concurrent.futures import ThreadPoolExecutor

from utils import make_dir_if_not_exists, download_image, fetch_all_urls_for_concept, save_yaml

kImagenetBaseUrl = "http://imagenet.stanford.edu/api/imagenet.synset.geturls?wnid="

def make_concepts_and_randoms(source_path, number_of_images_per_folder, number_of_random_folders, broden_dataset_path):
    print(f"Start creat data at {source_path}")
    if not os.path.exists(broden_dataset_path):
        raise Exception(f"Can not find broden dataset in path {broden_dataset_path}, please download broden dataset.")

    imagenet_target_classes = ['zebra']
    broden_concepts = ['striped', 'dotted', 'zigzagged']

    imagenet_dataframe = pd.read_csv("utils/imagenet_url_map.csv")
    imagenet_dataframe["url"] = kImagenetBaseUrl + imagenet_dataframe["synid"]

    # Make targets from imagenet
    for target_class in imagenet_target_classes:
        concept_path = os.path.join(source_path,target_class)
        make_dir_if_not_exists(concept_path)

        all_urls = []
        if imagenet_dataframe["class_name"].str.contains(target_class).any():
            # example: https://image-net.org/api/imagenet.synset.geturls?wnid=n01544389
            all_images = imagenet_dataframe[imagenet_dataframe["class_name"] == target_class]["url"].values[0]
            bytes = urllib.request.urlopen(all_images)
            for line in bytes:
                all_urls.append(line.decode("utf-8")[:-2])

        num_downloaded = 0
        pool = ThreadPoolExecutor()
        for image_url in all_urls:
            # We are filtering out images from Flickr urls, since several of those were removed
            if "flickr" not in image_url:
                try:
                    download_image(concept_path,image_url)
                    num_downloaded+=1
                except Exception as e:
                    print("Problem downloading imagenet image. Exception was " + str(e) + " for URL " + image_url)
        
            if num_downloaded >= number_of_images_per_folder:
                break

        # If we reached the end, notify the user through the console.
        if num_downloaded < number_of_images_per_folder:
            print("You requested " + str(number_of_images_per_folder) +
                " but we were only able to find " +
                str(num_downloaded) +
                " good images from imageNet for concept " + target_class)
        else:
            print("Downloaded " + str(number_of_images_per_folder) + " for " + target_class)

    # Make concepts from broden
    for concept in broden_concepts:
        concept_saving_path = os.path.join(source_path, concept)
        make_dir_if_not_exists(concept_saving_path)
        broden_textures_path = os.path.join(broden_dataset_path, "images/dtd/")

        for root,dirs, files in os.walk(broden_textures_path):
            texture_files = [a for a in files if (a.startswith(concept) and "color" not in a)]
            number_of_files_for_concept = len(texture_files)
            if number_of_images_per_folder > number_of_files_for_concept:
                raise Exception("Concept " + concept + " only contains " +
                                str(number_of_files_for_concept) +
                                " images. You requested " + str(number_of_images_per_folder))
            save_number = number_of_images_per_folder
            while save_number > 0:
                for file in texture_files:
                    path_file = os.path.join(root, file)
                    texture_saving_path_file = os.path.join(concept_saving_path, file)
                    shutil.copy(
                        path_file, texture_saving_path_file)  # change you destination dir
                    save_number -= 1
                    # Break if we saved all images
                    if save_number <= 0:
                        break

    # make random folders
    imagenet_concepts = imagenet_dataframe["class_name"].values.tolist()
    for partition_number in range(number_of_random_folders):
        partition_name = "random500_" + str(partition_number)
        partition_folder_path = os.path.join(source_path, partition_name)
        make_dir_if_not_exists(partition_folder_path)

        examples_selected = 0
        while examples_selected < number_of_images_per_folder:
            random_concept = random.choice(imagenet_concepts)
            urls = fetch_all_urls_for_concept(imagenet_dataframe, random_concept)
            for url in urls:
                # We are filtering out images from Flickr urls, since several of those were removed
                if "flickr" not in url:
                    try:
                        download_image(partition_folder_path, url)
                        examples_selected += 1
                        if (examples_selected) % 10 == 0:
                            print("Downloaded " + str(examples_selected) + "/" +
                                            str(number_of_images_per_folder) + " for " +
                                            partition_name)
                        break  # Break if we successfully downloaded an image
                    except:
                        pass # try new url

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create examples and concepts folders.')
    parser.add_argument('--source_dir', type=str, default="F:\\hierarchical_concept_explanation\\",
                        help='Name for the directory where we will create the data.')
    parser.add_argument('--number_of_images_per_folder', type=int, default=50,
                        help='Number of images to be included in each folder')
    parser.add_argument('--number_of_random_folders', type=int, default=3,
                        help='Number of folders with random examples that we will generate for tcav')
    parser.add_argument('--broden_dataset_path', type=str, default="F:\\Broden",
                        help='Number of folders with random examples that we will generate for tcav')
    parser.add_argument('--target_classes', type=list, default=['zebra'],
                        help='Number of folders with random examples that we will generate for tcav')
    parser.add_argument('--interested_concepts', type=list, default=['striped', 'dotted', 'zigzagged'],
                        help='Number of folders with random examples that we will generate for tcav')

    args = parser.parse_args()

    # source_dir: where images of concepts, target class and random images (negative samples when learning CAVs) live. Each should be a sub-folder within this directory. 
    make_dir_if_not_exists(args.source_dir)

    save_yaml(args.__dict__, "cav.yaml")

    make_concepts_and_randoms(args.source_dir,args.number_of_images_per_folder, args.number_of_random_folders, args.broden_dataset_path)
