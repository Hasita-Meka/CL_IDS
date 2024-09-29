import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import efficientnet_b0
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# Define Buffer class
class Buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.data = []
        self.labels = []
        self.logits = []

    def is_empty(self):
        return len(self.data) == 0

    def add_data(self, examples, labels, logits):
        self.data.extend(examples)
        self.labels.extend(labels)
        self.logits.extend(logits)

        if len(self.data) > self.buffer_size:
            self.data = self.data[-self.buffer_size:]
            self.labels = self.labels[-self.buffer_size:]
            self.logits = self.logits[-self.buffer_size:]

    def get_data(self, minibatch_size, device):
        idx = np.random.choice(len(self.data), minibatch_size, replace=False)
        buf_inputs = torch.stack([self.data[i] for i in idx]).to(device)
        buf_labels = torch.tensor([self.labels[i] for i in idx]).to(device)
        buf_logits = torch.stack([self.logits[i] for i in idx]).to(device)
        return buf_inputs, buf_labels, buf_logits


# Define constants
BATCH_SIZE = 1024  # Batch size for training
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Augmentation and Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dir = 'Dataset/train'
test_dir = 'Dataset/test'

# Load datasets
train_data = ImageFolder(train_dir, transform=transform)
test_data = ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Load EfficientNet model
model = efficientnet_b0(pretrained=True)
num_classes = len(train_data.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Modify the final layer
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


class Derpp:
    """Continual learning via Dark Experience Replay++."""

    def __init__(self, model, criterion, optimizer, buffer_size=1000, alpha=0.5, beta=0.5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.buffer = Buffer(buffer_size)  # Use the Buffer class
        self.alpha = alpha
        self.beta = beta

    def observe(self, inputs, labels, not_aug_inputs):
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        tot_loss = loss.item()

        # Dark Experience Replay
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(len(inputs), DEVICE)
            # MSE Loss on buffered logits
            buf_outputs = self.model(buf_inputs)
            loss_mse = self.alpha * nn.MSELoss()(buf_outputs, buf_logits)
            loss_mse.backward(retain_graph=True)  # Keep the graph for the next backward pass
            tot_loss += loss_mse.item()

            # Cross-Entropy Loss on buffered labels
            loss_ce = self.beta * self.criterion(buf_outputs, buf_labels)
            loss_ce.backward()  # No need to retain graph here
            tot_loss += loss_ce.item()

        self.optimizer.step()

        # Store the current inputs, labels, and logits in the buffer
        self.buffer.add_data(not_aug_inputs, labels.cpu().numpy(), outputs.data.cpu())

        return tot_loss


# Function to train the model using Derpp
def train_model_with_derpp(model, train_loader, criterion, optimizer, num_epochs, buffer_size=1000):
    derpp = Derpp(model, criterion, optimizer, buffer_size=buffer_size, alpha=0.5, beta=0.5)
    model.train()
    train_loss = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # Note: not_aug_inputs will be same as images for this implementation
            loss = derpp.observe(images, labels, images)
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return train_loss


# Function to evaluate the model (remains the same)
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    auc_roc = roc_auc_score(all_labels, all_preds, average='weighted', multi_class='ovr')
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, f1, precision, recall, auc_roc, cm


# Function to plot loss vs accuracy curves
def plot_metrics(train_loss, accuracy_list):
    epochs = range(1, len(train_loss) + 1)

    # Plotting Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_list, 'g', label='Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.show()


# Start timing
start_time = time.time()

# Train the model using Derpp
train_loss = train_model_with_derpp(model, train_loader, criterion, optimizer, NUM_EPOCHS)

# Evaluate the model
avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(model, test_loader)

# End timing
end_time = time.time()
time_taken = end_time - start_time

# Print performance metrics
print(f'Average Loss: {avg_loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'AUC-ROC: {auc_roc:.4f}')
print(f'Time Taken: {time_taken:.2f} seconds')
print('Confusion Matrix:\n', cm)

# Plot loss vs accuracy curves
accuracy_list = [accuracy] * NUM_EPOCHS  # For demonstration, using constant accuracy
plot_metrics(train_loss, accuracy_list)
