import gradio as gr
import os
import cv2
import numpy as np
from inference import run_inference
from PIL import Image
import tempfile
# Points color and marker
colors = [
    (255, 0, 0),  # L_lateral-incisor
    (0, 255, 0),  # L_canine
    (0, 0, 255),  # L_first-premolar
    (0, 255, 255),  # L_second-premolar
    (255, 255, 0),  # R_lateral incisor
    (255, 0, 255),  # R_Canine
    (128, 0, 128),  # R_first premolar
    (0, 165, 255)  # R_second premolar
]

color_dict = {
    'L_lateral-incisor': (255, 0, 0),
    'L_canine': (0, 255, 0),
    'L_first-premolar': (0, 0, 255),
    'L_second-premolar': (0, 255, 255),
    'R_lateral-incisor': (255, 255, 0),
    'R_Canine': (255, 0, 255),
    'R_first-premolar': (128, 0, 128),
    'R_second-premolar': (0, 165, 255)
}

classes = list(color_dict.keys())
markers = [1, 5]
available_models = [file.split(".")[0] for file in os.listdir("models/")]
default_model = available_models[0]

print(gr.__version__)

image_examples = [
    [os.path.join(os.path.dirname(__file__), "./images/1.jpg"), 0, []],
    [os.path.join(os.path.dirname(__file__), "./images/2.jpg"), 1, []],
    [os.path.join(os.path.dirname(__file__), "./images/3.JPG"), 2, []],
    [os.path.join(os.path.dirname(__file__), "./images/4.jpg"), 3, []],
    [os.path.join(os.path.dirname(__file__), "./images/5.jpg"), 4, []],
    [os.path.join(os.path.dirname(__file__), "./images/6.jpg"), 5, []]
]
custom_css = """
tbody {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
}
"""
# Application interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as app:
    with gr.Row():
        gr.Markdown(value="# Canine AI")
        with gr.Row():
            model_type = gr.Dropdown(["vit_h"], value='vit_h', label="Select Model", interactive=True)
            device = gr.Dropdown(
                choices=["cpu", "cuda"],
                value="cuda",
                label="Select Device", interactive=True
            )
    with gr.Accordion(label='Parameters', open=False):
        with gr.Row():
            points_per_side = gr.Number(value=32, label="points_per_side", precision=0,
                                        info='''The number of points to be sampled along one side of the image. The total 
                                        number of points is points_per_side**2.''', interactive=True)
            pred_iou_thresh = gr.Slider(value=0.88, minimum=0, maximum=1.0, step=0.01, label="pred_iou_thresh",
                                        info='''A filtering threshold in [0,1], using the model's predicted mask quality.''', interactive=True)
            stability_score_thresh = gr.Slider(value=0.95, minimum=0, maximum=1.0, step=0.01, label="stability_score_thresh",
                                               info='''A filtering threshold in [0,1], using the stability of the mask under 
                                               changes to the cutoff used to binarize the model's mask predictions.''', interactive=True)
            min_mask_region_area = gr.Number(value=0, label="min_mask_region_area", precision=0,
                                             info='''If >0, postprocessing will be applied to remove disconnected regions 
                                             and holes in masks with area smaller than min_mask_region_area.''', interactive=True)
        with gr.Row():
            stability_score_offset = gr.Number(value=1, label="stability_score_offset",
                                               info='''The amount to shift the cutoff when calculated the stability score.''', interactive=True)
            box_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="box_nms_thresh",
                                       info='''The box IoU cutoff used by non-maximal suppression to filter duplicate masks.''', interactive=True)
            crop_n_layers = gr.Number(value=0, label="crop_n_layers", precision=0,
                                      info='''If >0, mask prediction will be run again on crops of the image. 
                                      Sets the number of layers to run, where each layer has 2**i_layer number of image crops.''', interactive=True)
            crop_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="crop_nms_thresh",
                                        info='''The box IoU cutoff used by non-maximal suppression to filter duplicate 
                                        masks between different crops.''', interactive=True)
    with gr.Tab(label='Segementation'):
        with gr.Row().style(equal_height=True):
            with gr.Column():
                # Input image
                original_image = gr.State(value=None)   # Store original image without points, default None
                # Point prompt
                with gr.Tab("Original"):
                    input_image = gr.Image(type="numpy")
                    with gr.Column():
                        selected_points = gr.State([])      # Store points
                        with gr.Row():
                            # gr.Markdown('You can click on the image to select points prompt. Default: foreground_point.')
                            undo_button = gr.Button('Undo point')
                            reset_selected = gr.Button('Reset Image')
                        radio = gr.Radio(['foreground_point', 'background_point'], label='point labels', interactive=True)
                    # Run button
                
                    curr_class = gr.Dropdown(choices=classes,value="L_canine",interactive=True,label="Select Tooth")
                    button = gr.Button("Segment", interactive=True)
                
                
            # Show the image with mask
            with gr.Column():
                with gr.Tab(label='Image+Mask'):
                    output_image = gr.Image(type='numpy', show_download_button=True)
                # Show only mask
                with gr.Tab(label='Mask'):
                    output_mask = gr.Image(type='numpy', show_download_button=True)
                confirm_button = gr.Button("Confirm Segmentation", interactive=True)  # Add confirm button
                # undo_prev_seg = gr.Button("Undo previous Segmentation", interactive=True)
                reset_all = gr.Button("Reset All Segmentations", interactive=True)
                download_btn = gr.Button("Download Image with Mask", interactive=True)

        def process_example(img, ori_img, sel_p):
            return ori_img, []

        example = gr.Examples(
            examples=image_examples,
            inputs=[input_image, original_image, selected_points],
            outputs=[original_image, selected_points],
            fn=process_example,
            run_on_click=True
        )

    
    def store_img(img):
        return img, []  # When new image is uploaded, `selected_points` should be empty

    input_image.upload(
        store_img,
        [input_image],
        [original_image, selected_points]
    )

    # User click the image to get points, and show the points on the image
    def get_point(img, sel_pix, point_type, evt: gr.SelectData):
        if point_type == 'foreground_point':
            sel_pix.append((evt.index, 1))   # Append the foreground_point
        elif point_type == 'background_point':
            sel_pix.append((evt.index, 0))    # Append the background_point
        else:
            sel_pix.append((evt.index, 1))    # Default foreground_point
        # Draw points
        for point, label in sel_pix:
            cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        if img[..., 0][0, 0] == img[..., 2][0, 0]:  # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img if isinstance(img, np.ndarray) else np.array(img)
    
    input_image.select(
        get_point,
        [input_image, selected_points, radio],
        [input_image],
    )

    # Undo the selected point
    def undo_points(orig_img, sel_pix):
        if isinstance(orig_img, int):   # If orig_img is int, the image is selected from examples
            temp = cv2.imread(image_examples[orig_img][0])
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        else:
            temp = orig_img.copy()
        sel_pix.pop()  # Remove the last point
        # Draw remaining points
        for point, label in sel_pix:
            cv2.drawMarker(temp, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        return temp if isinstance(temp, np.ndarray) else np.array(temp), sel_pix
    
    undo_button.click(
        undo_points,
        [original_image, selected_points],
        [input_image, selected_points]
    )

    def reset_points(orig_img, sel_pnts):
        if isinstance(orig_img, int):   # If orig_img is int, the image is selected from examples
            temp = cv2.imread(image_examples[orig_img][0])
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        else:
            temp = orig_img.copy()

        return temp, []

    reset_selected.click(
        reset_points, 
        [original_image, selected_points],
        [input_image, selected_points]
    )


    # Button image
    prev_masks = gr.State([])
    chosen = gr.State([])

    def confirm_segmentation(img, orig_image, masks, prev_masks, curr_class, chosen):
        prev_masks.append((masks, curr_class))  # Append current mask and class to previous masks
        chosen.append(curr_class)
        new_options = [key for key in color_dict.keys() if key not in chosen]
        new_val = new_options[0] if len(new_options) > 0 else "No more classes left to segment"
        new_inter = True if new_val != -1 else False
        return orig_image, prev_masks, [], gr.Dropdown.update(choices=new_options, value=new_val, interactive=new_inter), chosen

    curr_masks = gr.State([])
    confirm_button.click(
        confirm_segmentation,
        [output_image, original_image, curr_masks, prev_masks, curr_class, chosen],
        [input_image, prev_masks, selected_points, curr_class, chosen]
    )


    def reset_all_masks(prev_masks, selected_pts, orig_img, curr_class):
        if isinstance(orig_img, int):   # If orig_img is int, the image is selected from examples
            temp = cv2.imread(image_examples[orig_img][0])
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        else:
            temp = orig_img.copy()

        return orig_img, [], gr.Image.update(value=None), gr.Image.update(value=None), gr.Dropdown.update(choices=classes, value='L_canine')
    
    reset_all.click(
        reset_all_masks,
        [prev_masks, selected_points, original_image, curr_class],
        [input_image, prev_masks, output_image, output_mask, curr_class]
    )

    def process_and_download_image(image):
    # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            pil_image.save(temp_file.name)
            return temp_file.name

    download_btn.click(
        process_and_download_image, 
        [output_image],
        [gr.File()]
        
    )
    button.click(run_inference, inputs=[device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh,
                                        min_mask_region_area, stability_score_offset, box_nms_thresh, crop_n_layers,
                                        crop_nms_thresh, original_image, selected_points, prev_masks, curr_class], show_progress="full",
                 outputs=[output_image, output_mask, original_image, curr_masks])

app.queue().launch(debug=True, share=True)
