import os
import argparse
import sys
from collections import defaultdict, deque
import pickle

import numpy as np
from PIL import Image
import cv2

from sahi.utils.coco import Coco
from sahi.utils.cv import get_bool_mask_from_coco_segmentation

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import segmentation_models_pytorch as smp

from transformers.models.maskformer.modeling_maskformer import dice_loss, sigmoid_focal_loss
import mlflow.pytorch
from mlflow import MlflowClient
import mlflow

import wandb




# Add the SAM directory to the system path
sys.path.append("./segment-anything")
from segment_anything import sam_model_registry

NUM_WORKERS = 4  # https://github.com/pytorch/pytorch/issues/42518
NUM_GPUS = torch.cuda.device_count()
DEVICE = 'cuda'


# Source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# Source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

# coco mask style dataloader
class Coco2MaskDataset(Dataset):
    def __init__(self, data_root, split, image_size):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        annotation = os.path.join(data_root, split, "_annotations.coco.json")
        self.coco = Coco.from_coco_dict_or_path(annotation)

        # TODO: use ResizeLongestSide and pad to square
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_resize = transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR)

    def __len__(self):
        return len(self.coco.images)

    def __getitem__(self, index):
        coco_image = self.coco.images[index]
        image = Image.open(os.path.join(self.data_root, self.split, coco_image.file_name)).convert("RGB")
        original_width, original_height = image.width, image.height
        ratio_h = self.image_size / image.height
        ratio_w = self.image_size / image.width
        image = self.image_resize(image)
        image = self.to_tensor(image)
        image = self.normalize(image)

        bboxes = []
        masks = []
        labels = []
        for annotation in coco_image.annotations:
            x, y, w, h = annotation.bbox
            # get scaled bbox in xyxy format
            bbox = [x * ratio_w, y * ratio_h, (x + w) * ratio_w, (y + h) * ratio_h]
            mask = get_bool_mask_from_coco_segmentation(annotation.segmentation, original_width, original_height)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.uint8)
            label = annotation.category_id
            bboxes.append(bbox)
            masks.append(mask)
            labels.append(label)
        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        labels = np.stack(labels, axis=0)
        return image, torch.tensor(bboxes), torch.tensor(masks).long()
    
    @classmethod
    def collate_fn(cls, batch):
        images, bboxes, masks = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, bboxes, masks

class SAMFinetuner(pl.LightningModule):

    def __init__(
            self,
            model_type,
            checkpoint_path,
            freeze_image_encoder=False,
            freeze_prompt_encoder=False,
            freeze_mask_decoder=False,
            batch_size=1,
            learning_rate=1e-4,
            weight_decay=1e-4,
            train_dataset=None,
            val_dataset=None,
            metrics_interval=10,
        ):
        super(SAMFinetuner, self).__init__()

        self.model_type = model_type
        self.model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)
        self.freeze_image_encoder = freeze_image_encoder
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_metric = defaultdict(lambda: deque(maxlen=metrics_interval))

        self.metrics_interval = metrics_interval

    
    def lr_scheduler_step(self, current_epoch, current_step=None, *args, **kwargs):
        warmup_steps = 250
        milestones = [0.66667, 0.86666]
        gamma = 0.1
    
        if current_step is not None:
            steps = current_step
        else:
            steps = current_epoch * self.trainer.estimated_steps_per_epoch
    
        if steps < warmup_steps:
            lr_scale = (steps + 1.) / float(warmup_steps)
        else:
            lr_scale = 1.
            for milestone in sorted(milestones):
                if steps >= milestone * self.trainer.estimated_steps_per_epoch:
                    lr_scale *= gamma
    
        return lr_scale


    def forward(self, imgs, bboxes, labels):
        _, _, H, W = imgs.shape
        features = self.model.image_encoder(imgs)
        num_masks = sum([len(b) for b in bboxes])

        loss_focal = loss_dice = loss_iou = 0.
        predictions = []
        tp, fp, fn, tn = [], [], [], []
        for feature, bbox, label in zip(features, bboxes, labels):
            # Embed prompts
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )
            # Predict masks
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=feature.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # Upscale the masks to the original image resolution
            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            predictions.append(masks)
            # Compute the iou between the predicted masks and the ground truth masks
            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                masks,
                label.unsqueeze(1),
                mode='binary',
                threshold=0.5,
            )
            batch_iou = smp.metrics.iou_score(batch_tp, batch_fp, batch_fn, batch_tn)
            # Compute the loss
            masks = masks.squeeze(1).flatten(1)
            label = label.flatten(1)
            loss_focal += sigmoid_focal_loss(masks, label.float(), num_masks)
            loss_dice += dice_loss(masks, label.float(), num_masks)
            loss_iou += F.mse_loss(iou_predictions, batch_iou, reduction='sum') / num_masks
            tp.append(batch_tp)
            fp.append(batch_fp)
            fn.append(batch_fn)
            tn.append(batch_tn)
        return {
            'loss': 20. * loss_focal + loss_dice + loss_iou,  # SAM default loss
            'loss_focal': loss_focal,
            'loss_dice': loss_dice,
            'loss_iou': loss_iou,
            'predictions': predictions,
            'tp': torch.cat(tp),
            'fp': torch.cat(fp),
            'fn': torch.cat(fn),
            'tn': torch.cat(tn),
        }
    
    def training_step(self, batch, batch_nb):
        imgs, bboxes, labels = batch
        outputs = self(imgs, bboxes, labels)

        for metric in ['tp', 'fp', 'fn', 'tn']:
            self.train_metric[metric].append(outputs[metric])

        # aggregate step metics
        step_metrics = [torch.cat(list(self.train_metric[metric])) for metric in ['tp', 'fp', 'fn', 'tn']]
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")
        metrics = {
            "loss": outputs["loss"],
            "loss_focal": outputs["loss_focal"],
            "loss_dice": outputs["loss_dice"],
            "loss_iou": outputs["loss_iou"],
            "train_per_mask_iou": per_mask_iou,
        }
        self.log_dict(metrics, prog_bar=True, rank_zero_only=True)
        mlflow.pytorch.autolog()
        wandb.log({"metrics": metrics})
        
        return metrics
    
    def validation_step(self, batch, batch_nb):
        imgs, bboxes, labels = batch
        outputs = self(imgs, bboxes, labels)
        outputs.pop("predictions")
        return outputs
    
    def validation_epoch_end(self, outputs):
        if NUM_GPUS > 1:
            outputs = all_gather(outputs)
            # the outputs are a list of lists, so flatten it
            outputs = [item for sublist in outputs for item in sublist]
        # aggregate step metics
        step_metrics = [
            torch.cat(list([x[metric].to(self.device) for x in outputs]))
            for metric in ['tp', 'fp', 'fn', 'tn']]
        # per mask IoU means that we first calculate IoU score for each mask
        # and then compute mean over these scores
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")

        metrics = {"val_per_mask_iou": per_mask_iou}
        self.log_dict(metrics)
        return metrics
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        def warmup_step_lr_builder(warmup_steps, milestones, gamma):
            def warmup_step_lr(steps):
                if steps < warmup_steps:
                    lr_scale = (steps + 1.) / float(warmup_steps)
                else:
                    lr_scale = 1.
                    for milestone in sorted(milestones):
                        if steps >= milestone * self.trainer.estimated_stepping_batches:
                            lr_scale *= gamma
                return lr_scale
            return warmup_step_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            warmup_step_lr_builder(250, [0.66667, 0.86666], 0.1)
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': "step",
                'frequency': 1,
            }
        }
    
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=self.val_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=False)
        return val_loader

# Set your variables directly
dataset_name = "data"
model_type = "vit_h"
checkpoint_path = "SAM/sam_vit_h_4b8939.pth"
freeze_image_encoder = True
freeze_prompt_encoder = False
freeze_mask_decoder = False
batch_size = 1
image_size = 1024
steps = 5
learning_rate = 1e-4
weight_decay = 1e-2
metrics_interval = 50
output_dir = "models"

# load the dataset
train_dataset = Coco2MaskDataset(data_root=dataset_name, split="train", image_size=image_size)
val_dataset = Coco2MaskDataset(data_root=dataset_name, split="val", image_size=image_size)

# create the model
model = SAMFinetuner(
    model_type,
    checkpoint_path,
    freeze_image_encoder=freeze_image_encoder,
    freeze_prompt_encoder=freeze_prompt_encoder,
    freeze_mask_decoder=freeze_mask_decoder,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    metrics_interval=metrics_interval,
)

callbacks = [
    LearningRateMonitor(logging_interval='step'),
    ModelCheckpoint(
        dirpath=output_dir,
        filename='{step}-{val_per_mask_iou:.2f}',
        save_last=True,
        save_top_k=1,
        monitor="val_per_mask_iou",
        mode="max",
        save_weights_only=True,
        every_n_train_steps=metrics_interval,
    ),
]

trainer = pl.Trainer(
    strategy='ddp' if NUM_GPUS > 2 else "deepspeed", #https://lightning.ai/docs/pytorch/stable/extensions/strategy.html
    accelerator=DEVICE,
    devices=NUM_GPUS,
    precision=32,
    callbacks=callbacks,
    max_epochs=-1, #-1
    max_steps=steps,
    val_check_interval=metrics_interval,
    check_val_every_n_epoch=None,
    num_sanity_val_steps=0,
)


mlflow.set_experiment('SAM Model')
# Auto log all MLflow entities
mlflow.pytorch.autolog(checkpoint=True)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="SAM-rtx3060",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "sam_vit_h",
    "dataset": "Breast",
    "esteps": steps,
    "batch_size": batch_size,
    "weight_decay": weight_decay,
    }
)

with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, "model")
    trainer.fit(model)
    wandb.finish()

mlflow.end_run()

