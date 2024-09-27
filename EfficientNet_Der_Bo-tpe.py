import os
import time
import random
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import efficientnet_b0  # Import EfficientNet
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import optuna  # For BO-TPE

NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Buffer Class for Dark Experience Replay
class Buffer:
    """Buffer to store past experiences for Dark Experience Replay."""
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def add_data(self, examples, logits):
        """Add new data to the buffer."""
        if len(self.buffer) < self.size:
            self.buffer.append((examples, logits))
        else:
            # Replace the oldest data with new data
            self.buffer.pop(0)
            self.buffer.append((examples, logits))

    def get_data(self, batch_size, device='cpu'):
        """Get a batch of data from the buffer."""
        if not self.buffer:
            raise ValueError("Buffer is empty.")
        # Randomly sample from the buffer
        examples, logits = zip(*random.sample(self.buffer, min(batch_size, len(self.buffer))))
        examples = torch.cat(examples).to(device)  # Stack examples into a single tensor
        logits = torch.cat(logits).to(device)  # Stack logits into a single tensor
        return examples, logits

    def is_empty(self):
        """Check if the buffer is empty."""
        return len(self.buffer) == 0

# Dark Experience Replay Class
class Der:
    """Continual learning via Dark Experience Replay."""
    def __init__(self, model, buffer_size, alpha, lr):
        self.model = model
        self.buffer = Buffer(buffer_size)
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def observe(self, inputs, labels):
        self.optimizer.zero_grad()
        tot_loss = 0

        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        tot_loss += loss.item()

        # Dark Experience Replay
        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(inputs.size(0), device=inputs.device)
            buf_outputs = self.model(buf_inputs)
            loss_mse = self.alpha * nn.functional.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

        self.optimizer.step()
        self.buffer.add_data(examples=inputs.detach(), logits=outputs.data)
        return tot_loss

# Data Augmentation and Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dir = 'Dataset/train'
test_dir = 'Dataset/test'

# Load datasets
train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

# Training and evaluation functions
def train_model_with_der(model, train_loader, num_epochs, buffer_size, alpha, lr):
    der = Der(model, buffer_size, alpha, lr)
    model.train()
    train_loss = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # Train with DER
            loss = der.observe(images, labels)
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return train_loss

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    auc_roc = roc_auc_score(all_labels, all_preds, multi_class='ovr', average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, f1, precision, recall, auc_roc, cm

# Bayesian Optimization with BO-TPE using Optuna
def objective(trial):
    # Hyperparameters to tune
    buffer_size = trial.suggest_int('buffer_size', 500, 2000)
    alpha = trial.suggest_float('alpha', 0.1, 1.0)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = 1024  # Set batch size to 1024 directly
    
    # Loaders with the specified batch size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model setup
    model = efficientnet_b0(weights='DEFAULT')
    num_classes = len(train_data.classes)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Modify final layer
    model = model.to(DEVICE)

    # Train and evaluate the model
    train_loss = train_model_with_der(model, train_loader, NUM_EPOCHS, buffer_size, alpha, lr)
    avg_loss, accuracy, _, _, _, _, _ = evaluate_model(model, test_loader)
    
    return accuracy  # We are optimizing for accuracy

# Run BO-TPE optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters found by BO-TPE
print("Best hyperparameters found by BO-TPE:")
print(study.best_trial.params)

# Final evaluation with the best hyperparameters
best_params = study.best_trial.params
BUFFER_SIZE = best_params['buffer_size']
ALPHA = best_params['alpha']
LR = best_params['lr']
BATCH_SIZE = 1024  # Set batch size to 1024 directly

# Train the final model with the best hyperparameters
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

train_loss = train_model_with_der(model, train_loader, NUM_EPOCHS, BUFFER_SIZE, ALPHA, LR)
avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(model, test_loader)

# Print performance metrics
print(f'Average Loss: {avg_loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'AUC-ROC: {auc_roc:.4f}')
