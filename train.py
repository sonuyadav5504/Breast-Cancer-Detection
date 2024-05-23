import torch
import os
import torchvision
import random
import supervision as sv
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch
from pytorch_lightning import Trainer
from pycocotools.coco import COCO
import cv2
import numpy as np
import matplotlib.pyplot as plt

from transformers import DetrImageProcessor
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")


dataset = '/kaggle/input/cv-ass-4-dataset'

a = os.path.join(dataset, "train2017")
b = os.path.join(dataset, "val2017")


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, 
        image_directory_path: str, 
        image_processor, 
        train: bool
    ):
        if train:
            annotation_file_path = "/kaggle/input/cv-ass-4-dataset/annotations/instances_train2017.json"
        else:
            annotation_file_path = "/kaggle/input/cv-ass-4-dataset/annotations/instances_val2017.json"

        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
    
train_dataset = CocoDetection(image_directory_path= a, image_processor=image_processor, train=True)
val_dataset = CocoDetection(image_directory_path= b, image_processor=image_processor, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))


# Visualize if dataset is loaded properly

# select random image
image_ids = train_dataset.coco.getImgIds()
image_id = random.choice(image_ids)
# image_id = 0
print('Image #{}'.format(image_id))

# load image and annotatons 
image = train_dataset.coco.loadImgs(image_id)[0]
annotations = train_dataset.coco.imgToAnns[image_id]
image_path = os.path.join(train_dataset.root, image['file_name'])
image = cv2.imread(image_path)

if len(annotations) > 0:

    # annotate
    detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
    categories = train_dataset.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}

    labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]

    box_annotator = sv.BoxAnnotator()
    frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

    %matplotlib inline  
    sv.show_frame_in_notebook(image, (8, 8))

else:
    print("There is no annoations for this image")

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

TRAIN_DATALOADER = DataLoader(dataset=train_dataset,num_workers =2, collate_fn=collate_fn, batch_size=8, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=val_dataset,num_workers=2, collate_fn=collate_fn, batch_size=4)

class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",num_labels= 1 , ignore_mismatched_sizes=True)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.avg_train_loss = 0
        self.avg_val_loss = 0
        self.train_losses = []
        self.val_losses = []

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict


    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("training_loss", loss, on_step=True, on_epoch=True)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_step=True, on_epoch=True)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
        return {"val_loss": loss}

    def on_train_epoch_end(self):
        self.avg_train_loss = self.trainer.callback_metrics["training_loss_epoch"].item()
        self.train_losses.append(self.avg_train_loss)
        self.log('avg_train_loss', self.avg_train_loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.avg_val_loss = self.trainer.callback_metrics["validation_loss_epoch"].item()
        self.val_losses.append(self.avg_val_loss)
        self.log('avg_val_loss', self.avg_val_loss, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

batch = next(iter(TRAIN_DATALOADER))

outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])


# settings
MAX_EPOCHS = 75

trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

trainer.fit(model)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(model.train_losses, label='Training loss')
plt.plot(model.val_losses, label='Validation loss')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Losses', fontsize=20)
plt.legend()
plt.show()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%
# MODEL_PATH = 'detr_model'
# model.model.save_pretrained(MODEL_PATH)

# # loading model
# model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
# model.to(DEVICE)

torch.save(model.state_dict(), 'CV_transformer.pt')

# Loading Model

# model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
model.load_state_dict(torch.load('CV_transformer.pt'))
model.to(DEVICE)

CONFIDENCE_TRESHOLD = 0.75

for i in range(10):
    
    categories = val_dataset.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}
    box_annotator = sv.BoxAnnotator()

    # select random image
    image_ids = val_dataset.coco.getImgIds()
    image_id = random.choice(image_ids)
    print('Image #{}'.format(image_id))

    # load image and annotatons 
    image = val_dataset.coco.loadImgs(image_id)[0]
    annotations = val_dataset.coco.imgToAnns[image_id]
    image_path = os.path.join(val_dataset.root, image['file_name'])
    image = cv2.imread(image_path)
    
    if len(annotations) > 0:

        # Annotate ground truth
        detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
        labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
        frame_ground_truth = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)


        # Annotate detections
        with torch.no_grad():

            # load image and predict
            inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
            outputs = model(**inputs)

            # post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
            results = image_processor.post_process_object_detection(
                outputs=outputs, 
                threshold=CONFIDENCE_TRESHOLD, 
                target_sizes=target_sizes
            )[0]


            detections = sv.Detections.from_transformers(transformers_results=results)
            labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
            frame_detections = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)


        # Combine both images side by side and display
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].imshow(cv2.cvtColor(frame_ground_truth, cv2.COLOR_BGR2RGB))
        axs[0].axis('off')
        axs[0].set_title('Ground Truth')

        axs[1].imshow(cv2.cvtColor(frame_detections, cv2.COLOR_BGR2RGB))
        axs[1].axis('off')
        axs[1].set_title('Detections')

        plt.show()
    else:
        print("Annotation for this image in available")
