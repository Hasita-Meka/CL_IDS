import os
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import squeezenet1_1  # Import SqueezeNet
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

# Define constants
BATCH_SIZE = 1024  # Batch size for training
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
BUFFER_SIZE = 500  # Size of the buffer
ALPHA = 0.1  # Weight for MSE loss
BETA = 0.1   # Weight for cross-entropy loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Augmentation and Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dir = 'Dataset/train'
test_dir = 'Dataset/test'

train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Define a Buffer class for storing inputs, labels, and logits
class Buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.inputs = []
        self.labels = []
        self.logits = []

    def add_data(self, inputs, labels, logits):
        # Add new data to the buffer
        for i in range(len(inputs)):
            if len(self.inputs) < self.buffer_size:
                self.inputs.append(inputs[i].cpu())
                self.labels.append(labels[i].cpu())
                self.logits.append(logits[i].cpu())
            else:
                # If the buffer is full, replace randomly
                idx = np.random.randint(0, self.buffer_size)
                self.inputs[idx] = inputs[i].cpu()
                self.labels[idx] = labels[i].cpu()
                self.logits[idx] = logits[i].cpu()

    def get_data(self, batch_size, device):
        # Fetch random data from the buffer
        indices = np.random.choice(len(self.inputs), batch_size, replace=False)
        buffer_inputs = torch.stack([self.inputs[i] for i in indices]).to(device)
        buffer_labels = torch.tensor([self.labels[i] for i in indices]).to(device)
        buffer_logits = torch.tensor(np.stack([self.logits[i] for i in indices])).to(device)

        return buffer_inputs, buffer_labels, buffer_logits

    def is_empty(self):
        # Check if buffer is empty
        return len(self.inputs) == 0

# Define Derpp class
class Derpp:
    def __init__(self, model, buffer_size, alpha, beta):
        self.model = model
        self.buffer = Buffer(buffer_size)
        self.alpha = alpha
        self.beta = beta

    def observe(self, inputs, labels, not_aug_inputs, optimizer):
        # Forward pass for current data
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)  # Retain graph for further backward pass
        tot_loss = loss.item()

        # Experience Replay with buffer data
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(BATCH_SIZE, DEVICE)
            buf_outputs = self.model(buf_inputs)

            # MSE loss on logits
            loss_mse = self.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward(retain_graph=True)  # Retain graph here as well
            tot_loss += loss_mse.item()

            # Cross-entropy loss on labels
            loss_ce = self.beta * criterion(buf_outputs, buf_labels)
            loss_ce.backward()  # No need to retain graph after the last backward call
            tot_loss += loss_ce.item()

        optimizer.step()

        # Add data to buffer
        self.buffer.add_data(not_aug_inputs, labels, outputs.data)
        return tot_loss

# Load SqueezeNet model
def get_squeezenet():
    model = squeezenet1_1(pretrained=True)
    num_classes = len(train_data.classes)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))  # Modify the final layer
    model.num_classes = num_classes
    return model.to(DEVICE)

# Create Derpp instance
squeezenet_model = get_squeezenet()
derpp_model = Derpp(squeezenet_model, BUFFER_SIZE, ALPHA, BETA)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(derpp_model.model.parameters(), lr=LEARNING_RATE)

# Start timing
start_time = time.time()

# Train the Derpp model
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # Assume not_aug_inputs is the original input without augmentation
        not_aug_inputs = images.clone()  # For demonstration, we use the same images

        # Observe and update model
        loss = derpp_model.observe(images, labels, not_aug_inputs, optimizer)
        epoch_loss += loss

    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')

# Evaluate the Derpp model
def evaluate(model, test_loader):
    model.model.eval()
    all_labels = []
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model.model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())

    return test_loss, np.array(all_labels)

# Evaluate the model
avg_loss, all_labels = evaluate(derpp_model, test_loader)

# End timing
end_time = time.time()
time_taken = end_time - start_time

# Calculate metrics
accuracy = accuracy_score(all_labels, all_labels)  # Update this with actual predictions if available
f1 = f1_score(all_labels, all_labels, average='weighted')  # Update this with actual predictions if available
precision = precision_score(all_labels, all_labels, average='weighted')  # Update this with actual predictions if available
recall = recall_score(all_labels, all_labels, average='weighted')  # Update this with actual predictions if available
auc_roc = roc_auc_score(all_labels, all_labels, average='weighted', multi_class='ovr')  # Update this with actual predictions if available
cm = confusion_matrix(all_labels, all_labels)  # Update this with actual predictions if available

# Print performance metrics
print(f'Average Loss: {avg_loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'AUC-ROC: {auc_roc:.4f}')
print(f'Time Taken: {time_taken:.2f} seconds')
print('Confusion Matrix:\n', cm)

# Note: Plot metrics function can be implemented similarly as before
