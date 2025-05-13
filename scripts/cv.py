import os
import time
from tempfile import TemporaryDirectory
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

# --- Configuration ---
DATA_DIR = #TODO: data_location  # Adjust this as needed
NUM_EPOCHS = 25
BATCH_SIZE = 4
NUM_WORKERS = 4

def setup_device():
    """Detects and sets up the available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[CHECKPOINT] Using device: {device}")
    if device.type == "cuda":
        print(f"[CHECKPOINT] CUDA device: {torch.cuda.get_device_name(0)}")
    return device

def load_data(data_dir, batch_size, num_workers, pin_memory):
    """Loads training and validation datasets with transformations."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers,
                                       pin_memory=pin_memory)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"[CHECKPOINT] Loaded data: {len(class_names)} classes ({class_names})")
    print(f"[CHECKPOINT] Training set size: {dataset_sizes['train']}, Validation set size: {dataset_sizes['val']}")

    return dataloaders, dataset_sizes, class_names

def build_model(num_classes, device):
    """Builds a ResNet18 model adapted to the number of classes."""
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    print(f"[CHECKPOINT] Model built: ResNet18 with {num_classes} output classes")
    return model

def setup_optimizer(model):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    print("[CHECKPOINT] Optimizer and scheduler set up.")
    return optimizer, scheduler

def train_model(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, num_epochs):
    """Handles the training and validation phases for the model."""
    since = time.time()
    best_acc = 0.0

    with TemporaryDirectory() as tempdir:  # Safe temp dir for saving best model weights
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if dataset_sizes.get(phase, 0) == 0:
                    print(f"[WARNING] Skipping {phase} phase: no data found.")
                    continue

                model.train() if phase == 'train' else model.eval()
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train' and scheduler is not None:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best validation Acc: {best_acc:.4f}')

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
        permanent_save_dir = #TODO: model_save_location
        os.makedirs(permanent_save_dir, exist_ok=True)  # Ensure directory exists

        permanent_save_path = os.path.join(permanent_save_dir, 'best_resnet18_model.pt')
        torch.save(model.state_dict(), permanent_save_path)
        print(f"[CHECKPOINT] Best model permanently saved to {permanent_save_path}")
    return model


def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def save_input_grid(inputs, class_names, labels):
    """Save a denormalized image grid from training batch to results directory."""
    # Where to save
    save_dir = #TODO: saved_image_location
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'batch_image_grid.png')

    # Denormalize the inputs so they look normal (un-gray)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inputs_denorm = denormalize(inputs.clone(), mean, std)
    inputs_denorm.clamp_(0, 1)

    # Make and save the image grid
    grid = torchvision.utils.make_grid(inputs_denorm)
    torchvision.utils.save_image(grid, save_path)
    print(f"[CHECKPOINT] Saved batch image grid to {save_path}")

    # Print class labels in the batch
    print("[SUMMARY] Labels in this batch:")
    for idx in labels:
        print(f" - {class_names[idx]}")

if __name__ == '__main__':
    cudnn.benchmark = True

    device = setup_device()
    pin_memory = device.type == 'cuda'

    dataloaders, dataset_sizes, class_names = load_data(DATA_DIR, BATCH_SIZE, NUM_WORKERS, pin_memory)
    model = build_model(len(class_names), device)
    optimizer, scheduler = setup_optimizer(model)
    criterion = nn.CrossEntropyLoss()

    print("[START] Beginning training loop...")
    trained_model = train_model(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, NUM_EPOCHS)
    print("[COMPLETE] Training finished.")
    inputs, labels = next(iter(dataloaders['train']))
    save_input_grid(inputs, class_names, labels)
