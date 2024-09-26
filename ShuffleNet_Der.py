import os
import time
import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import shufflenet_v2_x0_5
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from torchvision import datasets, transforms

# Constants
BATCH_SIZE = 1024
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = 1000
ALPHA = 0.5

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

# Load ShuffleNet model
model = shufflenet_v2_x0_5(pretrained=True)
num_classes = len(train_data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Buffer class for experience replay
class Buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.inputs = []
        self.logits = []

    def add_data(self, examples, logits):
        for i in range(len(examples)):
            if len(self.inputs) < self.buffer_size:
                self.inputs.append(examples[i].cpu())
                self.logits.append(logits[i].cpu())
            else:
                idx = np.random.randint(0, self.buffer_size)
                self.inputs[idx] = examples[i].cpu()
                self.logits[idx] = logits[i].cpu()

    def get_data(self, batch_size, device):
        indices = np.random.choice(len(self.inputs), batch_size, replace=False)
        buffer_inputs = torch.stack([self.inputs[i] for i in indices]).to(device)
        buffer_logits = torch.tensor(np.stack([self.logits[i] for i in indices])).to(device)
        return buffer_inputs, buffer_logits

    def is_empty(self):
        return len(self.inputs) == 0

# Der class for continual learning
class Der:
    def __init__(self, model, buffer_size, alpha, optimizer):
        self.model = model
        self.buffer = Buffer(buffer_size)
        self.alpha = alpha
        self.optimizer = optimizer

    def observe(self, inputs, labels, not_aug_inputs):
        self.optimizer.zero_grad()
        tot_loss = 0

        # Forward pass
        outputs = self.model(inputs)

        # Compute the primary loss
        loss = criterion(outputs, labels)
        loss.backward()
        tot_loss += loss.item()

        # Experience Replay
        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(BATCH_SIZE, DEVICE)
            buf_outputs = self.model(buf_inputs)

            # Compute MSE loss for buffered logits
            loss_mse = self.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

        # Update the model parameters
        self.optimizer.step()

        # Add new data to buffer
        self.buffer.add_data(not_aug_inputs, outputs.data)

        return tot_loss

# Training Function
def train_model(model, train_loader, optimizer, num_epochs, der_model):
    model.train()
    train_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Der observation
            loss = der_model.observe(images, labels, images)
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return train_loss

# Instantiate Der model
der_model = Der(model=model, buffer_size=BUFFER_SIZE, alpha=ALPHA, optimizer=optimizer)

# Evaluation function
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

# Start timing
start_time = time.time()

# Train the model using Der
train_loss = train_model(model, train_loader, optimizer, NUM_EPOCHS, der_model)

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
accuracy_list = [accuracy] * NUM_EPOCHS
plt.plot(train_loss, label="Training Loss")
plt.plot(accuracy_list, label="Accuracy")
plt.legend()
plt.show()
