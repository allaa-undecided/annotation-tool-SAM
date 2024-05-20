from PIL import Image, ImageDraw  # Import PIL for image handling and drawing
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor  # Import SAM models
import gc  # Import garbage collector for memory management
import os  # Import OS module for file and directory operations
import numpy as np  # Import NumPy for numerical operations
import torch  # Import PyTorch for machine learning tasks
import cv2  # Import OpenCV for image processing
import gradio as gr  # Import Gradio for building the web interface

# Dictionary of available models and their checkpoint paths
models = {
    'vit_b': './checkpoints/vit_b.pth',
    'vit_l': './checkpoints/vit_l.pth',
    'vit_h': './checkpoints/vit_h.pth'
}

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

# Example images for the interface
image_examples = [
    [os.path.join(os.path.dirname(__file__), "./images/1.jpg"), 0, []],
    [os.path.join(os.path.dirname(__file__), "./images/2.jpg"), 1, []],
    [os.path.join(os.path.dirname(__file__), "./images/3.JPG"), 2, []],
    [os.path.join(os.path.dirname(__file__), "./images/4.jpg"), 3, []],
    [os.path.join(os.path.dirname(__file__), "./images/5.jpg"), 4, []],
    [os.path.join(os.path.dirname(__file__), "./images/6.jpg"), 5, []]
]

# Function to draw bounding boxes on an image
def plot_boxes(img, boxes):
    img_pil = Image.fromarray(np.uint8(img * 255)).convert('RGB')  # Convert NumPy array to PIL Image
    draw = ImageDraw.Draw(img_pil)  # Create a drawing context
    for box in boxes:  # Draw each bounding box
        color = tuple(np.random.randint(0, 255, size=3)).tolist()  # Random color for each box
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)  # Convert coordinates to integers
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)  # Draw the rectangle
    return img_pil  # Return the image with drawn boxes

# Function to segment an image and generate masks
def segment_one(img, mask_generator, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set seed for reproducibility
    masks = mask_generator.generate(img)  # Generate masks using the mask generator
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)  # Sort masks by area
    mask_all = np.ones((img.shape[0], img.shape[1], 3))  # Create an array to hold all masks
    for ann in sorted_anns:  # Apply each mask to the array
        m = ann['segmentation']
        color_mask = np.random.random((1, 3)).tolist()[0]  # Random color for each mask
        for i in range(3):
            mask_all[m == True, i] = color_mask[i]
    result = img / 255 * 0.3 + mask_all * 0.7  # Blend the original image with the masks
    return result, mask_all  # Return the result and the mask array

# Function to run inference using the generator
def generator_inference(device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh,
                        min_mask_region_area, stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thresh,
                        input_x, progress=gr.Progress()):
    # Load the SAM model
    sam = sam_model_registry[model_type](checkpoint=models[model_type]).to(device)
    # Initialize the mask generator
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=1,
        point_grids=None,
        min_mask_region_area=min_mask_region_area,
        output_mode='binary_mask'
    )

    # If input is an image (NumPy array)
    if type(input_x) == np.ndarray:
        result, mask_all = segment_one(input_x, mask_generator)  # Segment the image
        return result, mask_all
    elif isinstance(input_x, str):  # If input is a video (path string)
        cap = cv2.VideoCapture(input_x)  # Read the video
        frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc('x', '2', '6', '4'), fps, (W, H), isColor=True)
        while True:
            ret, frame = cap.read()  # Read a frame
            if ret:
                result, mask_all = segment_one(frame, mask_generator, seed=2023)  # Segment the frame
                result = (result * 255).astype(np.uint8)  # Convert to uint8
                out.write(result)  # Write the frame to the output video
            else:
                break
        out.release()
        cap.release()
        return 'output.mp4'  # Return the path to the output video

# Function to run inference using the predictor
def predictor_inference(device, model_type, input_x, input_text, selected_points, owl_vit_threshold=0.1, curr_class=None, prev_masks=None):
    # Load the SAM model
    sam = sam_model_registry[model_type](checkpoint=models[model_type]).to(device)
    predictor = SamPredictor(sam)  # Initialize the predictor
    predictor.set_image(input_x)  # Process the image to produce an image embedding

    transformed_boxes = None

    # Process selected points
    if len(selected_points) != 0:
        points = torch.Tensor([p for p, _ in selected_points]).to(device).unsqueeze(1)
        labels = torch.Tensor([int(l) for _, l in selected_points]).to(device).unsqueeze(1)
        transformed_points = predictor.transform.apply_coords_torch(points, input_x.shape[:2])
        print(points.size(), transformed_points.size(), labels.size(), input_x.shape, points)
    else:
        transformed_points, labels = None, None

    # Predict segmentation based on points
    masks, scores, logits = predictor.predict_torch(
        point_coords=transformed_points,
        point_labels=labels,
        boxes=transformed_boxes,  # Only one box
        multimask_output=False,
    )
    masks = masks.cpu().detach().numpy()
    mask_all = np.ones((input_x.shape[0], input_x.shape[1], 3))
    
    # Apply masks to the image
    for ann in masks:
        mask = ann[0]  # Extract the single mask
        if not curr_class:
            color_mask = np.random.random((3,)).tolist()
        else:
            color_mask = [c / 255.0 for c in color_dict[curr_class]]
        for i in range(3):
            mask_all[mask == True, i] = color_mask[i]
    
    img = input_x / 255 * 0.3 + mask_all * 0.7  # Blend the original image with the masks
    for prev, clr_class in prev_masks:
        for idk in prev:
            mask = idk[0]
            mask = np.array(mask)
            color_mask = [c / 255.0 for c in color_dict[clr_class]]
            for i in range(3):
                 mask_all[mask == True, i] = color_mask[i]
    
    img = input_x / 255 * 0.3 + mask_all * 0.7  # Final blend
    
    # Clean up
    del input_text
    gc.collect()
    torch.cuda.empty_cache()
    return img, mask_all, input_x, masks  # Return results

# Function to run inference based on the input type and user selections
def run_inference(device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area,
                  stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thresh, input_x,
                  selected_points=[], prev_masks=[], curr_class="L_canine"):
    input_text = ''
    owl_vit_threshold = 0.1
    if isinstance(input_x, int):  # If input_x is int, the image is selected from examples
        input_x = cv2.imread(image_examples[input_x][0])
        input_x = cv2.cvtColor(input_x, cv2.COLOR_BGR2RGB)
    if (input_text != '' and not isinstance(input_x, str)) or len(selected_points) != 0:  # User input text or points
        print('use predictor_inference')
        print('prompt points length: ', len(selected_points))
        return predictor_inference(device, model_type, input_x, input_text, selected_points, owl_vit_threshold, curr_class=curr_class, prev_masks=prev_masks)
    else:
        print('use generator_inference')
        return generator_inference(device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh,
                                   min_mask_region_area, stability_score_offset, box_nms_thresh, crop_n_layers,
                                   crop_nms_thresh, input_x)
