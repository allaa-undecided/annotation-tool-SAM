# Canine Segmentation and Annotation Tool
This is gradio WebUI specialized for teeth segmentation based on **[this repository](https://github.com/5663015/segment_anything_webui)**, but extends it to allow for masks to presist after several segmentations, with added functionality for different classes/annotations of teeth.
It allows the use of different **[SAM models](https://github.com/facebookresearch/segment-anything)** by Meta, on both CPU and GPU, to segment and annotate teeth for Canine Impaction measurements.
This repo was made for my CS undergraduate Thesis.

## Usage

- First, start by creating a conda environment:

```
    conda create -n <env_name> python=3.9
```

- Activate environment then install required packagaes from requirements file:

```
    conda activate <env_name>
    pip install -r requirements.txt
```

- Download the different SAM models checkpoints into checkpoints folder and rename them from their respective links as follows:

    - `vit_h`: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

    - `vit_l`: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)

    - `vit_b`: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
    
```
    mkdir checkpoints
    cd checkpoints
    wget <link>
```
    
- Run using (default model is "vit_h" using GPU):

```
    python app.py
```
    
## Expected Usage Behavior

- First, select the model from the dropdown list, and CPU or CUDA based on whether you have a GPU or not.
- Next, upload an image.
- Start by adding points on the image for a **SINGLE** tooth you want to segment. (Foreground points are chosen by default, Background points are ones that tell the model to ignore a specific part.)
- Choose the label of the tooth you want to apply.
- Click segment. 
- The original image with segmented tooth should appear, if you are happy with it, you can confirm segmentation to save the mask. If not, you can undo points and place different ones and segment again till you are happy with the results.
 
## References
- Awesome person who first made this **[this repository](https://github.com/5663015/segment_anything_webui)**
