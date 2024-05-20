import gradio as gr  # Import Gradio for building the web interface
import os  # Import OS module for file and directory operations
import cv2  # Import OpenCV for image processing
import numpy as np  # Import NumPy for numerical operations
from inference import run_inference  # Import the custom inference function
from PIL import Image  # Import PIL for image handling
import tempfile  # Import tempfile for temporary file creation

# Points color and marker definitions
colors = [
    (255, 0, 0),  # L_lateral-incisor (Red)
    (0, 255, 0),  # L_canine (Green)
    (0, 0, 255),  # L_first-premolar (Blue)
    (0, 255, 255),  # L_second-premolar (Cyan)
    (255, 255, 0),  # R_lateral incisor (Yellow)
    (255, 0, 255),  # R_Canine (Magenta)
    (128, 0, 128),  # R_first premolar (Purple)
    (0, 165, 255)  # R_second premolar (Orange)
]

# Dictionary mapping tooth classes to their respective colors
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

# List of tooth classes
classes = list(color_dict.keys())
# Marker types for each class
markers = [1, 5]
# List available models from the 'models' directory
available_models = [file.split(".")[0] for file in os.listdir("models/")]
# Set default model to the first one in the list
default_model = available_models[0]

print(gr.__version__)  # Print Gradio version

# Example images for the interface
image_examples = [
    [os.path.join(os.path.dirname(__file__), "./images/1.jpg"), 0, []],
    [os.path.join(os.path.dirname(__file__), "./images/2.jpg"), 1, []],
    [os.path.join(os.path.dirname(__file__), "./images/3.JPG"), 2, []],
    [os.path.join(os.path.dirname(__file__), "./images/4.jpg"), 3, []],
    [os.path.join(os.path.dirname(__file__), "./images/5.jpg"), 4, []],
    [os.path.join(os.path.dirname(__file__), "./images/6.jpg"), 5, []]
]

# Custom CSS for the interface
custom_css = """
tbody {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
}
"""

# Create the Gradio app interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as app:
    with gr.Row():
        gr.Markdown(value="# Canine AI")  # Title of the app
        with gr.Row():
            model_type = gr.Dropdown(["vit_h"], value='vit_h', label="Select Model", interactive=True)  # Model selection dropdown
            device = gr.Dropdown(  # Device selection dropdown
                choices=["cpu", "cuda"],
                value="cuda",
                label="Select Device", interactive=True
            )
    
    # Accordion for parameter settings
    with gr.Accordion(label='Parameters', open=False):
        with gr.Row():
            points_per_side = gr.Number(value=32, label="points_per_side", precision=0,  # Number of points per side
                                        info='''The number of points to be sampled along one side of the image. The total 
                                        number of points is points_per_side**2.''', interactive=True)
            pred_iou_thresh = gr.Slider(value=0.88, minimum=0, maximum=1.0, step=0.01, label="pred_iou_thresh",  # IoU threshold
                                        info='''A filtering threshold in [0,1], using the model's predicted mask quality.''', interactive=True)
            stability_score_thresh = gr.Slider(value=0.95, minimum=0, maximum=1.0, step=0.01, label="stability_score_thresh",  # Stability score threshold
                                               info='''A filtering threshold in [0,1], using the stability of the mask under 
                                               changes to the cutoff used to binarize the model's mask predictions.''', interactive=True)
            min_mask_region_area = gr.Number(value=0, label="min_mask_region_area", precision=0,  # Minimum mask region area
                                             info='''If >0, postprocessing will be applied to remove disconnected regions 
                                             and holes in masks with area smaller than min_mask_region_area.''', interactive=True)
        with gr.Row():
            stability_score_offset = gr.Number(value=1, label="stability_score_offset",  # Stability score offset
                                               info='''The amount to shift the cutoff when calculated the stability score.''', interactive=True)
            box_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="box_nms_thresh",  # Box NMS threshold
                                       info='''The box IoU cutoff used by non-maximal suppression to filter duplicate masks.''', interactive=True)
            crop_n_layers = gr.Number(value=0, label="crop_n_layers", precision=0,  # Number of crop layers
                                      info='''If >0, mask prediction will be run again on crops of the image. 
                                      Sets the number of layers to run, where each layer has 2**i_layer number of image crops.''', interactive=True)
            crop_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="crop_nms_thresh",  # Crop NMS threshold
                                        info='''The box IoU cutoff used by non-maximal suppression to filter duplicate 
                                        masks between different crops.''', interactive=True)
    
    # Segmentation tab
    with gr.Tab(label='Segmentation'):
        with gr.Row().style(equal_height=True):
            with gr.Column():
                # Input image
                original_image = gr.State(value=None)   # Store original image without points, default None
                # Point prompt
                with gr.Tab("Original"):
                    input_image = gr.Image(type="numpy")
                    with gr.Column():
                        selected_points = gr.State([])  # Store selected points
                        with gr.Row():
                            undo_button = gr.Button('Undo point')  # Button to undo last point
                            reset_selected = gr.Button('Reset Image')  # Button to reset image
                        radio = gr.Radio(['foreground_point', 'background_point'], label='point labels', interactive=True)  # Radio button for point labels
                    # Run button
                    curr_class = gr.Dropdown(choices=classes, value="L_canine", interactive=True, label="Select Tooth")  # Dropdown to select tooth class
                    button = gr.Button("Segment", interactive=True)  # Segment button
                
            # Column to show image with mask and mask only
            with gr.Column():
                with gr.Tab(label='Image+Mask'):
                    output_image = gr.Image(type='numpy', show_download_button=True)  # Image with mask
                with gr.Tab(label='Mask'):
                    output_mask = gr.Image(type='numpy', show_download_button=True)  # Mask only
                confirm_button = gr.Button("Confirm Segmentation", interactive=True)  # Confirm segmentation button
                reset_all = gr.Button("Reset All Segmentations", interactive=True)  # Reset all segmentations button
                download_btn = gr.Button("Download Image with Mask", interactive=True)  # Download image with mask button

        # Function to process example images
        def process_example(img, ori_img, sel_p):
            return ori_img, []

        # Example images
        example = gr.Examples(
            examples=image_examples,
            inputs=[input_image, original_image, selected_points],
            outputs=[original_image, selected_points],
            fn=process_example,
            run_on_click=True
        )

    # Function to store uploaded image
    def store_img(img):
        return img, []  # When new image is uploaded, `selected_points` should be empty

    # Upload event for input image
    input_image.upload(
        store_img,
        [input_image],
        [original_image, selected_points]
    )

    # Function to get point from image click event
    def get_point(img, sel_pix, point_type, evt: gr.SelectData):
        if point_type == 'foreground_point':
            sel_pix.append((evt.index, 1))   # Append the foreground_point
        elif point_type == 'background_point':
            sel_pix.append((evt.index, 0))    # Append the background_point
        else:
            sel_pix.append((evt.index, 1))    # Default foreground_point
        # Draw points on the image
        for point, label in sel_pix:
            cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        if img[..., 0][0, 0] == img[..., 2][0, 0]:  # BGR to RGB conversion
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img if isinstance(img, np.ndarray) else np.array(img)
    
    # Image click event to get points
    input_image.select(
        get_point,
        [input_image, selected_points, radio],
        [input_image],
    )

    # Function to undo the last selected point
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
        if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB conversion
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        return temp if isinstance(temp, np.ndarray) else np.array(temp), sel_pix
    
    # Click event for undo button
    undo_button.click(
        undo_points,
        [original_image, selected_points],
        [input_image, selected_points]
    )

    # Function to reset selected points
    def reset_points(orig_img, sel_pnts):
        if isinstance(orig_img, int):   # If orig_img is int, the image is selected from examples
            temp = cv2.imread(image_examples[orig_img][0])
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        else:
            temp = orig_img.copy()
        return temp, []

    # Click event for reset selected button
    reset_selected.click(
        reset_points, 
        [original_image, selected_points],
        [input_image, selected_points]
    )

    # State to store previous masks and chosen classes
    prev_masks = gr.State([])
    chosen = gr.State([])

    # Function to confirm segmentation and store masks
    def confirm_segmentation(img, orig_image, masks, prev_masks, curr_class, chosen):
        prev_masks.append((masks, curr_class))  # Append current mask and class to previous masks
        chosen.append(curr_class)
        new_options = [key for key in color_dict.keys() if key not in chosen]  # Update options for remaining classes
        new_val = new_options[0] if len(new_options) > 0 else "No more classes left to segment"
        new_inter = True if new_val != -1 else False
        return orig_image, prev_masks, [], gr.Dropdown.update(choices=new_options, value=new_val, interactive=new_inter), chosen

    curr_masks = gr.State([])

    # Click event for confirm button
    confirm_button.click(
        confirm_segmentation,
        [output_image, original_image, curr_masks, prev_masks, curr_class, chosen],
        [input_image, prev_masks, selected_points, curr_class, chosen]
    )

    # Function to reset all segmentations
    def reset_all_masks(prev_masks, selected_pts, orig_img, curr_class):
        if isinstance(orig_img, int):   # If orig_img is int, the image is selected from examples
            temp = cv2.imread(image_examples[orig_img][0])
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        else:
            temp = orig_img.copy()
        return orig_img, [], gr.Image.update(value=None), gr.Image.update(value=None), gr.Dropdown.update(choices=classes, value='L_canine')
    
    # Click event for reset all button
    reset_all.click(
        reset_all_masks,
        [prev_masks, selected_points, original_image, curr_class],
        [input_image, prev_masks, output_image, output_mask, curr_class]
    )

    # Function to process and download image
    def process_and_download_image(image):
        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            pil_image.save(temp_file.name)
            return temp_file.name

    # Click event for download button
    download_btn.click(
        process_and_download_image, 
        [output_image],
        [gr.File()]
    )

    # Click event for segment button
    button.click(run_inference, inputs=[device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh,
                                        min_mask_region_area, stability_score_offset, box_nms_thresh, crop_n_layers,
                                        crop_nms_thresh, original_image, selected_points, prev_masks, curr_class], show_progress="full",
                 outputs=[output_image, output_mask, original_image, curr_masks])

# Launch the Gradio app
app.queue().launch(debug=True, share=True)
