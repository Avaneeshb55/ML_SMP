# MNIST Digit Classifier Complete Project (All 6 Tasks)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# =========================================================
#     TASK-1         Understand the Dataset
# MNIST has:
#   - 60,000 training images, 10,000 test images
#   - 28x28 grayscale images
#   - 10 classes (digits 0 through 9)
# =========================================================

def explore_dataset():
    raw_data=datasets.MNIST(root='./data',train=True,download=True)

    print("=== Dataset Info ===")
    print(f"Training samples : {len(raw_data)}")
    print(f"Image size       : {raw_data[0][0].size}")
    print(f"Number of classes: 10  (digits 0-9)")

    fig,axes=plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle("Sample MNIST Images",fontsize=14)
    for i,ax in enumerate(axes.flat):
        image,label=raw_data[i]
        ax.imshow(image,cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("sample_images.png")
    plt.show()
    print("Sample images saved as sample_images.png\n")


# ================================================================================================================
#               TASK-2          Data Pipeline
# Normalize the data 
#  transforms.ToTensor()  converts the values of pixels from (0,255) to (0,1)
#  We use the MNIST mean and variance of 0.1307 and 0.3081
#  transforms.Normalize((0.1307,), (0.3081,))  this converts each pixel value to (-0.424,2.821)  (By Z scoring)
# =================================================================================================================

def get_dataloaders(batch_size=64):
    train_transform=transforms.Compose([
        transforms.RandomRotation(degrees=10),                                        # RandomRotation to rotate the digits slightly
        transforms.RandomAffine(degrees=0,translate=(0.1,0.1),scale=(0.9,1.1)),       # Used RandomAffine for small shifts and zoom
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset= datasets.MNIST(root='./data',train=True, download=True,transform=train_transform)
    test_dataset= datasets.MNIST(root='./data',train=False,download=True,transform=test_transform)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    images, labels= next(iter(train_loader))
    print("="*15 + "Data Pipeline" + "="*15)
    print(f"One batch shape: images={images.shape}, labels={labels.shape}\n")

    return train_loader, test_loader, test_dataset


# =========================================================
#      TASK-3          CNN Architecture
# =========================================================

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1= nn.Linear(64 * 7 * 7, 128)
        self.fc2  = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28x28 to 14x14
        x = self.pool(self.relu(self.conv2(x)))   # 14x14 to  7x7
        x = self.relu(self.conv3(x))              # 7x7  to  7x7
        x = x.view(x.size(0),-1)                 # flatten to 3136
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)                           # logits 
        return x


# =========================================================
#   TASK-4                Training Loop
# =========================================================

def train(model, train_loader,optimizer,criterion,epoch,device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx,(images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()               # 1. clear old gradients
        outputs = model(images)             # 2. forward pass
        loss = criterion(outputs, labels)   # 3. compute loss
        loss.backward()                     # 4. backpropagation
        optimizer.step()                    # 5. update weights

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct+= (predicted == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 150 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = (100.0*correct)/ total
    print(f"  Train Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.2f}%")
    return avg_loss,accuracy


# =========================================================
#      TASK-5         Evaluation
# =========================================================

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device),labels.to(device)
            outputs = model(images)
            loss = criterion(outputs,labels)
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct+= (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    print(f"   Test  Loss: {avg_loss:.4f} | Test  Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


# =========================================================
#            TASK 6: Save, Load and Inference
# =========================================================

def save_model(model, path="mnist_cnn_weights.pt"):
    torch.save(model.state_dict(), path)
    print(f"\nModel saved to {path}")


def load_model(path="mnist_cnn_weights.pt"):
    model = CNN()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    print(f"Model loaded from {path}")
    return model


def run_inference_grid(model, test_loader, device):
    images, labels = next(iter(test_loader))
    images_gpu = images.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images_gpu)
        predictions = outputs.argmax(dim=1).cpu()

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Model Predictions on Test Images", fontsize=14)
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        color = 'green' if predictions[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {predictions[i].item()}  True: {labels[i].item()}", color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("predictions_grid.png")
    plt.show()
    print("Prediction grid saved as predictions_grid.png")


def run_inference_single(model, test_dataset, device, index=0):
    """Shows a single image with its probability bar chart (like the screenshot)."""
    image_tensor, true_label = test_dataset[index]

    model.eval()
    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(device)  
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        predicted = probs.argmax()
        confidence = probs[predicted] * 100

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))

    
    ax_img.imshow(image_tensor.squeeze().numpy(), cmap='gray')
    color = 'green' if predicted == true_label else 'red'
    ax_img.set_title(
        f"Prediction: {predicted}  (conf: {confidence:.1f}%)\nTrue label: {true_label}",
        color=color, fontsize=13, fontweight='bold'
    )
    ax_img.axis('off')

    
    bar_colors = ["#4B90DA"]*10
    bar_colors[predicted] ='#E05C5C'   
    ax_bar.bar(range(10), probs, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax_bar.set_xticks(range(10))
    ax_bar.set_xlabel('Digit class', fontsize=11)
    ax_bar.set_ylabel('Probability', fontsize=11)
    ax_bar.set_ylim([0, 1])
    ax_bar.set_title('Class Probabilities', fontsize=12)

    plt.tight_layout()
    plt.savefig("inference_single.png", dpi=150)
    plt.show()
    print(f"Single inference saved as inference_single.png")
    print(f"Predicted: {predicted} | True: {true_label} | Confidence: {confidence:.2f}%")


def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies, num_epochs):
   
    epochs = range(1, num_epochs + 1)
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    fig.suptitle('MNIST  Training Curves', fontsize=14, fontweight='bold')
 
    #======== Plot 1: Loss =========
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:red', linestyle='--', marker='o')
    ax1.plot(epochs, test_losses, label='Test Loss', color='darkred', linestyle='-', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(list(epochs))
 
    # ========= Plot 2: Accuracy ==========
    ax2.plot(epochs, train_accuracies, label='Train Accuracy', color='tab:blue', linestyle='--', marker='s')
    ax2.plot(epochs, test_accuracies, label='Test Accuracy',color='darkblue', linestyle='-', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(list(epochs))
 
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Training curves saved as training_curves.png")


# =========== MAIN =============

def main():
    print("=" * 50)
    print("  MNIST Digit Classifier ")
    print("=" * 50)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("No GPU found, running on CPU\n")

    # explore dataset
    explore_dataset()

    #  data loaders
    train_loader,test_loader,test_dataset = get_dataloaders(batch_size=64)

    #  model
    model = CNN().to(device)
    print("="*15 + "Model Architecture" + "="*15)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_params:,}\n")

    # train and evaluate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    num_epochs = 10
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies  = []

    print("="*15 + "Training" + "="*15)
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        scheduler.step()

    # plots
    plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies, num_epochs)

    # save model
    save_model(model, "mnist_cnn_weights.pt")

    # inference: grid of 10 images
    run_inference_grid(model, test_loader, device)

    # inference: single image with probability bar chart
    run_inference_single(model, test_dataset, device, index=0)

    # Final result
    print("\n" + "="*15 + "Final Result" + "="*15)
    print(f"Best Test Accuracy: {max(test_accuracies):.2f}%")


if __name__ == "__main__":
    main()