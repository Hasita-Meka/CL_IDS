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
import optuna  # Added for hyperparameter optimization

# Constants
BATCH_SIZE = 1024
NUM_EPOCHS = 30
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
def create_model(learning_rate):
    model = shufflenet_v2_x0_5(pretrained=True)
    num_classes = len(train_data.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer


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

    def observe(self, inputs, labels, not_aug_inputs, criterion):  # Add criterion here
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
def train_model(model, train_loader, optimizer, num_epochs, der_model, criterion):  # Add criterion here
    model.train()
    train_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Der observation
            loss = der_model.observe(images, labels, images, criterion)  # Pass criterion
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return train_loss



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


# BO-TPE optimization function
def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)  # Suggest learning rates
    model, criterion, optimizer = create_model(learning_rate)

    # Instantiate Der model
    der_model = Der(model=model, buffer_size=BUFFER_SIZE, alpha=ALPHA, optimizer=optimizer)

    # Train the model
    train_loss = train_model(model, train_loader, optimizer, NUM_EPOCHS, der_model, criterion)  # Pass criterion

    # Evaluate the model
    avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(model, test_loader)

    return accuracy  # Return the metric to optimize



# Start optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # Number of trials can be adjusted

# Best hyperparameters found
print("Best hyperparameters found: ", study.best_params)

# Retrieve the best model's metrics for final evaluation
best_learning_rate = study.best_params['learning_rate']
best_model, _, _ = create_model(best_learning_rate)

# Final evaluation of the best model
final_train_loss = train_model(best_model, train_loader, optimizer, NUM_EPOCHS, der_model)
final_avg_loss, final_accuracy, final_f1, final_precision, final_recall, final_auc_roc, final_cm = evaluate_model(
    best_model, test_loader)

# Print final performance metrics
print(f'Final Average Loss: {final_avg_loss:.4f}')
print(f'Final Accuracy: {final_accuracy:.4f}')
print(f'Final F1 Score: {final_f1:.4f}')
print(f'Final Precision: {final_precision:.4f}')
print(f'Final Recall: {final_recall:.4f}')
print(f'Final AUC-ROC: {final_auc_roc:.4f}')
print('Final Confusion Matrix:\n', final_cm)

# Plot loss vs accuracy curves for final model
final_accuracy_list = [final_accuracy] * NUM_EPOCHS
plt.plot(final_train_loss, label="Training Loss")
plt.plot(final_accuracy_list, label="Final Accuracy")
plt.legend()
plt.show()
