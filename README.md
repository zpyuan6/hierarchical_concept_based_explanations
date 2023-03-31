# hierarchical_concept_based_explanations

## 1. How to use

You could use following codes to prepare env.
```
conda create -n h_concept_explanation python=3.
conda activate h_concept_explanation
pip install -r requirements.txt
```

### 1.1 Dataset preparation
1. Dataset download
    - [Broden dataset download](https://github.com/CSAILVision/NetDissect)
        ```
        wget --progress=bar http://netdissect.csail.mit.edu/data/broden1_227.zip -O broden1_227.zip
        ```
    - [ImageNet dataset download](https://www.image-net.org/index.php) 
        You are required to download ILSVRC2012_devkit_t12.tar.gz for getting class_to_id dictionary for model

2. Data folder making
    As shown in [TCAV start code](https://github.com/tensorflow/tcav/blob/b922c44bcc64c6bdddb8f661d732fa2145c99d95/Run_TCAV.ipynb), you are required to make a data folder (***source_dir***) including
    - images for each concept from Broden dataset,
    - images for the class/labels of interest from ImageNet dataset,
    - random images that will be negative examples when learning CAVs (images that probably don't belong to any concepts) from ImageNet dataset.
    You could run the following code to create data folder automatically, or create it manually. Manual creation may be quicker due to network or dataset updates. Detailed requirements for data folder can be found in [TCAV start code](https://github.com/tensorflow/tcav/blob/b922c44bcc64c6bdddb8f661d732fa2145c99d95/Run_TCAV.ipynb).
    ```
    python utils/download_and_make_datasets.py
    ```
### 1.2 Model preparation

### 1.3 

### 1.4 

## 2. To do list

## 3. Related works