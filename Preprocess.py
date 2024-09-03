import os
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
import torch.nn as nn
from monai.transforms import (
    Spacingd,
    Orientationd,
    Compose,
    Resized,
    ToTensord,
    ScaleIntensityRanged,
    Lambdad,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd
)
from collections import Counter

# Function to load NIfTI images and convert to numpy array
def load_nifti(file_path):
    return nib.load(file_path).get_fdata()

# Function to pad images to the same size
def pad_image(image, target_height, target_width):
    if len(image.shape) == 4:  # Case for 4D tensor
        _, _, h, w = image.shape
    elif len(image.shape) == 3:  # Case for 3D tensor
        _, h, w = image.shape
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    pad_h = target_height - h
    pad_w = target_width - w
    if len(image.shape) == 4:
        padded_image = F.pad(image, (0, pad_w, 0, pad_h), 'constant', 0)
    else:
        padded_image = F.pad(image, (0, pad_w, 0, pad_h), 'constant', 0)
    return padded_image

# Function to get file list excluding hidden files
def get_file_list(directory):
    return [f for f in os.listdir(directory) if not f.startswith('.')]

# Function to preprocess the data
def preprocess_data(image_files, label_files, transforms, data_dir, label_dir):
    preprocessed_data = []
    for image_file, label_file in zip(image_files, label_files):
        # Load the original image and label
        original_image = load_nifti(os.path.join(data_dir, image_file))
        label_image = load_nifti(os.path.join(label_dir, label_file))

        # Extract the FLAIR image (assuming it's the first channel, index 0)
        FLAIR_image = original_image[..., 0]  # Adjust the index if necessary

        # Convert to dictionary format for MONAI transforms
        data = {"image": np.expand_dims(FLAIR_image, axis=0), "label": np.expand_dims(label_image, axis=0)}

        # Apply preprocessing
        data = transforms(data)

        # Convert to numpy arrays
        image = data["image"].cpu().numpy()
        label = data["label"].cpu().numpy()

        preprocessed_data.append((image, label))
    return preprocessed_data



# Define preprocessing transformations for training and test
train_transforms = Compose([
    Orientationd(keys=["image", "label"], axcodes='RAS'),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["image", "label"], spatial_size=(240, 240, 155), mode=('trilinear', 'nearest')),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
#     RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
#     RandGaussianNoised(keys=["image"], prob=0.5),
    Lambdad(keys=["label"], func=lambda x: x.astype(np.int32)),  # Convert labels to int32
    ToTensord(keys=["image", "label"])
])

test_transforms = Compose([
    Orientationd(keys=["image", "label"], axcodes='RAS'),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["image", "label"], spatial_size=(240, 240, 155), mode=('trilinear', 'nearest')),
    Lambdad(keys=["label"], func=lambda x: x.astype(np.int32)),  # Convert labels to int32
    NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
    ToTensord(keys=["image", "label"])
])

# Function to count class frequencies in label files
def count_class_frequencies(label_files, label_dir):
    class_counts = Counter()
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        label_data = load_nifti(label_path)
        unique, counts = np.unique(label_data, return_counts=True)
        class_counts.update(dict(zip(unique, counts)))
    return class_counts

# Function to calculate class weights
def calculate_class_weights(class_counts):
    total_pixels = sum(class_counts.values())
    class_weights = {cls: total_pixels / count for cls, count in class_counts.items()}
    
    # Normalize weights (optional)
    max_weight = max(class_weights.values())
    class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}
    
    return class_weights
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_losss
