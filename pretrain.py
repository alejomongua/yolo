import time

import scipy.io
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_

from torchvision import datasets, transforms

from yolo import TinyYOLO as YOLO

NUMBER_OF_CLASSES = 20  # 20 classes from VOC dataset
GRID_SIZE = 7  # Grid size (e.g., 7 x 7)
NUMBER_OF_BBOXES = 2  # Number of bounding boxes per grid cell (e.g., 3)
IMAGE_SIZE = 448

TRAIN_FOLDER_PATH = '/data/maestria/datasets/imagenet'
VAL_FOLDER_PATH = '/data/maestria/datasets/imagenet_val'
GROUND_TRUTH_PATH = '/data/maestria/datasets/ILSVRC2012_validation_ground_truth.txt'
META_FILE_PATH = '/data/maestria/datasets/meta.mat'


class ImageNetClassifier(nn.Module):
    def __init__(self, yolo_model, num_classes=1000):  # ImageNet has 1000 classes
        super(ImageNetClassifier, self).__init__()

        # Use the convolutional layers from YOLO
        self.layer1 = yolo_model.layer1
        self.layer2 = yolo_model.layer2
        self.layer3 = yolo_model.layer3
        self.layer4 = yolo_model.layer4
        self.layer5 = yolo_model.layer5
        self.layer6 = yolo_model.layer6
        self.layer7 = yolo_model.layer7

        # Global Average Pooling (GAP) Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer
        # Assuming the last layer's channel size is 1024
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Pass through YOLO layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        # Global Average Pooling and Classification
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.fc(x)
        return x


def validate(model, validation_loader, loss_function, device):
    """
    Validation step during training.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    # Initialize tqdm progress bar
    progress_bar = tqdm(validation_loader, desc="Validating", leave=False)

    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()

            # Update the progress bar
            progress_bar.set_postfix(
                val_loss=total_loss / len(validation_loader))

    return total_loss / len(validation_loader)


def train_model(model, train_loader, validation_loader, num_epochs=10):
    """
    Main training loop for the model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device '{device}'.")

    # Define the loss function with the appropriate parameters
    loss_function = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.1)

    # Transfer the model to the GPU
    model.to(device)

    # Initialize lists to track losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        # Use tqdm for progress bar
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for images, targets in progress_bar:
            # Transfer images and targets to the GPU
            images = images.to(device)
            targets = targets.to(device)

            # Zero the gradients on each iteration
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate the loss
            loss = loss_function(outputs, targets)

            # Backward pass
            loss.backward()

            # Clip gradients to avoid exploding gradients
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

            # Update the progress bar
            progress_bar.set_postfix(train_loss=total_loss / len(train_loader))

        # Calculate the average loss for the epoch
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate the model
        avg_val_loss = validate(
            model, validation_loader, loss_function, device)
        val_losses.append(avg_val_loss)

        # Update the learning rate scheduler
        scheduler.step(avg_val_loss)

        # Print the loss for the epoch
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Save the model if it has the best validation loss so far
        if epoch == 0 or avg_val_loss < min(val_losses[:-1]):
            torch.save(model.state_dict(), 'imagenet_model_best.pth')

    # Plot the training and validation losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model


def create_class_idx_mapping(meta_file):
    meta = scipy.io.loadmat(meta_file)
    synsets = meta['synsets']
    wnid_to_idx = {m[0][1][0]: m[0][0][0][0] -
                   1 for m in synsets}  # -1 for 0-based index
    return wnid_to_idx


def get_imagenet_train_loader(train_dir, wnid_to_idx, batch_size=64, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

    # Update class_to_idx mapping in ImageFolder
    train_dataset.class_to_idx = wnid_to_idx

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader


class ImageNetValidationDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load ground truth labels
        self.labels = list(
            map(int, open(annotation_file).read().strip().split('\n')))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir, f'ILSVRC2012_val_{idx + 1:08d}.JPEG')
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx] - 1  # Adjust index to 0-based

        if self.transform:
            image = self.transform(image)

        return image, label


def get_imagenet_val_loader(val_dir, ground_truth, batch_size=64, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    val_dataset = ImageNetValidationDataset(root_dir=val_dir, annotation_file=ground_truth,
                                            transform=transform)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return val_loader


wnid_to_idx = create_class_idx_mapping(META_FILE_PATH)

imagenet_loader_train = get_imagenet_train_loader(
    TRAIN_FOLDER_PATH, wnid_to_idx, batch_size=64, image_size=IMAGE_SIZE)

imagenet_loader_val = get_imagenet_val_loader(
    VAL_FOLDER_PATH, GROUND_TRUTH_PATH, batch_size=64, image_size=IMAGE_SIZE)


yolo_model = YOLO(NUMBER_OF_CLASSES)
model = ImageNetClassifier(yolo_model, num_classes=1000)
model = train_model(model, imagenet_loader_train, imagenet_loader_val)

model_name = f'imagenet_model_{int(time.time())}.pth'
torch.save(model.state_dict(), model_name)
