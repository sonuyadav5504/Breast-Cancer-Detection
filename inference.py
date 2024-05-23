import os
import torch
import cv2
import pytorch_lightning as pl
from torchvision.ops import nms
from transformers import DetrForObjectDetection, DetrImageProcessor

image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def convert_to_yolo_format(boxes, image_width, image_height):
    yolo_boxes = []
    for box in boxes:
        # Calculate center x and y
        x_center = (box[0] + box[2]) / 2.0
        y_center = (box[1] + box[3]) / 2.0
        
        # Calculate width and height
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        # Normalize bounding box coordinates
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height
        
        yolo_boxes.append([x_center, y_center, width, height])

    return yolo_boxes

# Define a function to apply NMS
def apply_nms(boxes, scores, iou_threshold=0.1):

    boxes = boxes.clone().detach().to(device) 
    scores = scores.clone().detach().to(device)


    # Apply non-maximum suppression
    keep = nms(boxes, scores, iou_threshold)

    # Filter detections
    filtered_boxes = [boxes[i] for i in keep]
    filtered_scores = [scores[i] for i in keep]

    return filtered_boxes, filtered_scores

def test_model(model, image_folder, output_folder, device):
    # Set the model to evaluation mode
    model.eval()

    # Load all images in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        
        # Process the image
        encoding = image_processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"]
        pixel_mask = encoding['pixel_mask']
        
        with torch.no_grad():
            # Load image and predict
            inputs = {
                "pixel_values": pixel_values.to(device),
                "pixel_mask": pixel_mask.to(device)
            }
            outputs = model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(device)
            results = image_processor.post_process_object_detection(
                outputs=outputs, 
                threshold=0.01, 
                target_sizes=target_sizes
            )[0]
            
            # Extract bounding boxes and scores
            boxes = results['boxes']
            scores = results['scores']
            
            # Apply NMS
            boxes, scores = apply_nms(boxes, scores)

            # Get image width and height
            image_height, image_width, _ = image.shape
            
            # Convert bounding boxes to YOLO format
            yolo_boxes = convert_to_yolo_format(boxes, image_width, image_height)
            
            # Write predictions to text file
            output_file = os.path.splitext(image_file)[0] + "_preds.txt"
            output_path = os.path.join(output_folder, output_file)
            with open(output_path, 'w') as f:
                for box, score in zip(yolo_boxes, scores):
                    # Format: center_x center_y width height confidence_score
                    f.write(' '.join(map(lambda x: f"{x:.18e}", box)) + ' ' + f"{score:.18e}" + '\n')


class Detr(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",num_labels=1,ignore_mismatched_sizes=True)

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = Detr()
model.load_state_dict(torch.load('CV_transformer.pt'))

model.to(device)
model.eval()

# Define image folder and output folder
image_folder = "D:\\Sem 2\\CV\\Assignment_4\\sample_test\\test\\images"
output_folder = "D:\\Sem 2\\CV\\Assignment_4\\trash_output"

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Perform inference on images and save predictions
test_model(model, image_folder, output_folder, device)
