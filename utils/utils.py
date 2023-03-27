import os
import yaml
import urllib.request
from PIL import Image

def load_yaml(yaml_path):
  with open(yaml_path,"r") as f:
    content = f.read()
  yamlData = yaml.load(content, yaml.FullLoader)
  # print("load yaml parameters, yaml type:", type(yamlData))
  # print(yamlData)
  return yamlData

def save_yaml(data, save_path):
  with open(save_path,"w") as f:
    yaml.dump(data,f)

def make_dir_if_not_exists(path):
  if not os.path.exists(path):
    os.makedirs(path)

""" Downloads an image.
Downloads and image from a image url provided and saves it under path.
Filters away images that are corrupted or smaller than 10KB
  Args:
    path: Path to the folder where we're saving this image.
    url: url to this image.
  Raises:
    Exception: Propagated from PIL.image.verify()
"""
def download_image(path, url):
  image_name = url.split("/")[-1]
  image_name = image_name.split("?")[0]
  image_prefix = image_name.split(".")[0]
  saving_path = os.path.join(path, image_prefix + ".jpg")
  urllib.request.urlretrieve(url, saving_path)

  try:
    # Throw an exception if the image is unreadable or corrupted
    Image.open(saving_path).verify()

    # Remove images smaller than 10kb, to make sure we are not downloading empty/low quality images
    if os.stat(saving_path).st_size < 10000:
      os.remove(saving_path)
      raise Exception("images smaller than 10kb")
  # PIL.Image.verify() throws a default exception if it finds a corrupted image.
  except Exception as e:
    os.remove(
        saving_path
    )  # We need to delete it, since urllib automatically saves them.
    raise e

""" For a imagenet label, fetches all URLs that contain this image, from the main URL contained in the dataframe
  Args:
    imagenet_dataframe: Pandas Dataframe containing the URLs for different
      imagenet classes.
    concept: A string representing Imagenet concept(i.e. "zebra").
  Returns:
    A list containing all urls for the imagenet label. For example
    ["abc.com/image.jpg", "abc.com/image2.jpg", ...]
  Raises:
    tf.errors.NotFoundError: Error occurred when we can't find the imagenet
    concept on the dataframe.
"""
def fetch_all_urls_for_concept(imagenet_dataframe, concept):
  if imagenet_dataframe["class_name"].str.contains(concept).any():
    all_images = imagenet_dataframe[imagenet_dataframe["class_name"] ==
                                    concept]["url"].values[0]
    bytes = urllib.request.urlopen(all_images)
    all_urls = []
    for line in bytes:
      all_urls.append(line.decode("utf-8")[:-2])
    return all_urls
  else:
    raise Exception("Couldn't find any imagenet concept for " + concept +
        ". Make sure you're getting a valid concept")