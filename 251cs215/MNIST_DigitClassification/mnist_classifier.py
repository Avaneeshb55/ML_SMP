# MNIST Digit Classifier

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ========================================================
# TASK 1: Understand the Dataset
# MNIST has:
#   - 60,000 training images, 10,000 test images
#   - 28x28 grayscale images
#   - 10 classes (digits 0 through 9)
# =========================================================

def explore_dataset():
    raw_data = datasets.MNIST(root='./data', train=True, download=True)

    print("=== Dataset Info ===")
    print(f"Training samples : {len(raw_data)}")
    print(f"Image size : {raw_data[0][0].size}")
    print(f"Number of classes : 10  (digits 0-9)")

    # Visualize a few samples
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle("Sample MNIST Images", fontsize=14)

    for i, ax in enumerate(axes.flat):
        image, label = raw_data[i]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("sample_images.png")
    plt.show()
    print("Sample images saved as sample_images.png\n")


# ===============================================================================================================================================
# TASK 2: Data Pipeline
# We normalize images so pixel values go from [0, 255] to roughly [-1, 1]. This helps the model learn faster.
# mean=0.1307 and std=0.3081 are the known values for MNIST.

# DATA AUGMENTATION (only applied during training):
# The idea is to artificially create slightly different versions of each image so the model sees more variety and generalizes better.
# We use small, realistic augmentations because digits like 6 and 9 can flip into each other if you rotate by 180 degrees
#   RandomRotation(10)     - rotate the digit slightly (10 degrees)
#   RandomAffine           - small shifts and zoom to simulate handwriting variation
# ================================================================================================================================================

def get_dataloaders(batch_size=64):
    # Training transforms , includes augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),             # rotate ±10 degrees
        transforms.RandomAffine(
            degrees=0,                                     # no extra rotation here
            translate=(0.1, 0.1),                          # shift up to 10% in x/y
            scale=(0.9, 1.1)                               # zoom in/out slightly
        ),
        transforms.ToTensor(),                             # convert to tensor (0-1 range)
        transforms.Normalize((0.1307,), (0.3081,))        # normalize using MNIST mean & std
    ])

    # Test transforms , no augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    images, labels = next(iter(train_loader))
    print("=== Data Pipeline ===")
    print(f"One batch shape: images={images.shape}, labels={labels.shape}")
    print()

    return train_loader, test_loader


# ===========================================
# TASK 3: CNN Architecture
# ===========================================

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


# =================================
# TASK 4: Training Loop
# =================================

def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()

    total_loss = 0
    correct    = 0
    total      = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to the same device as the model (GPU or CPU)
        images, labels = images.to(device), labels.to(device)

        # Step 1: clear old gradients
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Track stats
        total_loss += loss.item()
        predicted   = outputs.argmax(dim=1)
        correct    += (predicted == labels).sum().item()
        total      += labels.size(0)

        # Print progress every 150 batches
        if (batch_idx + 1)%150 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    print(f"  Train Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


# ===============================
# TASK 5: Evaluation
# ===============================

def evaluate(model, test_loader, criterion, device):
    model.eval() 

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to the same device as the model (GPU or CPU)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total+= labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    print(f"  → Test  Loss: {avg_loss:.4f} | Test  Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


# ========================================
# TASK 6: Save, Load & Inference
# ========================================

def save_model(model, path="mnist_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"\nModel saved to {path}")


def load_model(path="mnist_model.pth"):
    model = MyCNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model


def run_inference(model, test_loader, device):
    """Show a few test images and what the model predicted."""
    images, labels = next(iter(test_loader))

    # Move to device for inference
    images_gpu = images.to(device)

    model.eval()
    with torch.no_grad():
        outputs     = model(images_gpu)
        predictions = outputs.argmax(dim=1).cpu()   # move predictions back to CPU for display

    # Un-normalize the images just for display
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Model Predictions on Test Images", fontsize=14)

    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        color = 'green' if predictions[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {predictions[i].item()}  True: {labels[i].item()}", color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.show()
    print("Predictions image saved as predictions.png")






def main():
    print("=" * 50)
    print("  MNIST Digit Classifier — CNN with PyTorch")
    print("=" * 50)
    print()

    # Detect device uses GPU if available, otherwise falls back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("No GPU found, running on CPU\n")

    # Task 1 : explore the dataset
    explore_dataset()

    # Task 2 : set up data loaders
    train_loader, test_loader = get_dataloaders(batch_size=64)

    # Task 3 : create the model and move it to the device
    model = MyCNN().to(device)   # .to(device) moves all weights to GPU/CPU
    print("=== Model Architecture ===")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_params:,}\n")

    # Task 4 & 5 : train and evaluate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler: gradually reduce LR over epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    num_epochs = 10
    train_accuracies = []
    test_accuracies  = []

    print("=== Training ===")
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        scheduler.step()

    # Plot training history
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(range(1, num_epochs + 1), test_accuracies,  label='Test Accuracy',  marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_curve.png")
    plt.show()

    # Task 6 : save the model and run inference
    save_model(model)
    run_inference(model, test_loader, device)

    print("\n=== Final Result ===")
    print(f"Best Test Accuracy: {max(test_accuracies):.2f}%")

if __name__ == "__main__":
    main()