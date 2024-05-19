# Canine Segmentation and Annotation Tool
This is gradio WebUI specialized for teeth segmentation based on **[this repository](https://github.com/5663015/segment_anything_webui)**.
It allows the use of different **[SAM models]((https://segment-anything.com/))** by Meta, on both CPU and GPU, to segment and annotate teeth for Canine Impaction measurements.
This repo was made for my CS undergraduate Thesis.

## Usage

- First, start by creating a conda environment:

'''
    conda create -n <env_name> python=3.9
'''

- Activate environment then install required packagaes from requirements file:

'''
    conda activate <env_name>
    pip install -r requirements.txt
'''

- Download the different SAM models checkpoints into checkpoints folder and rename them from their respective links as follows:

    - `vit_h`: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

    - `vit_l`: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)

    - `vit_b`: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
    
'''
    mkdir checkpoints
    cd checkpoints
    wget <link>

'''
