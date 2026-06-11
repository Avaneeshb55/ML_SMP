import torch
import torch.nn as nn
from torchvision import datasets, transforms
import random

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 =nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3= nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.pool= nn.MaxPool2d(2,2)
        self.dropout= nn.Dropout(0.25)
        self.fc1= nn.Linear(64*7*7,128)
        self.fc2= nn.Linear(128, 10)
        self.relu= nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


model= CNN()
model.load_state_dict(torch.load('mnist_cnn_weights.pt',map_location='cpu'))
model.eval()


test_dataset = datasets.MNIST('./data',train=False,download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
                ]))


idx = random.randint(0, 9999)
image, true_label= test_dataset[idx]
print("=" * 50)
print("  MNIST Digit Classifier ")
print("=" * 50)
print(f"Image index: {idx}")


with torch.no_grad():
    output = model(image.unsqueeze(0))
    prediction = output.argmax(dim=1).item()

print(f"Predicted : {prediction}")
print(f"True label: {true_label}")

if (prediction == true_label) :
    print(f"Prediction is CORRECT !")
else:
    print(f"Prediction is WRONG!")
    

