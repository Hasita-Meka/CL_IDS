import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import squeezenet1_1  # Import SqueezeNet
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import optuna  # Import Optuna for hyperparameter optimization

# Define constants
BATCH_SIZE = 1024  # Batch size for training
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
train_data = ImageFolder(train_dir, transform=transform)
test_data = ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.CrossEntropyLoss() 

# Load SqueezeNet model
model = squeezenet1_1(pretrained=True)
num_classes = len(train_data.classes)
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))  # Modify the final layer
model.num_classes = num_classes
model = model.to(DEVICE)

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

# Function to train the model using DER
def train_model(model, train_loader, optimizer, num_epochs, der_model):
    model.train()
    train_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # DER observation
            loss = der_model.observe(images, labels, images)
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return train_loss

# Function to evaluate the model
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

# Optuna objective function for hyperparameter optimization
def objective(trial):
    # Suggest hyperparameters to optimize
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 16, 1024, step=16)

    # Create a DataLoader with the suggested batch size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize model and optimizer with suggested hyperparameters
    model = squeezenet1_1(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
    model.num_classes = num_classes
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Instantiate DER model
    der_model = Der(model=model, buffer_size=BUFFER_SIZE, alpha=ALPHA, optimizer=optimizer)

    # Train the model
    train_loss = train_model(model, train_loader, optimizer, NUM_EPOCHS, der_model)

    # Evaluate the model
    avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(model, test_loader)

    # Return the accuracy as the objective metric to maximize
    return accuracy

# Start timing
start_time = time.time()

# Run the optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print best hyperparameters
print("Best hyperparameters: ", study.best_params)

# Best model evaluation
best_learning_rate = study.best_params['learning_rate']
best_batch_size = study.best_params['batch_size']

# Use best hyperparameters to retrain the model
train_loader = DataLoader(train_data, batch_size=best_batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=best_batch_size, shuffle=False)

model = squeezenet1_1(pretrained=True)
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
model.num_classes = num_classes
model = model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=best_learning_rate)

# Instantiate DER model
der_model = Der(model=model, buffer_size=BUFFER_SIZE, alpha=ALPHA, optimizer=optimizer)

# Train the model using the best hyperparameters
train_loss = train_model(model, train_loader, optimizer, NUM_EPOCHS, der_model)

# Evaluate the best model
avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(model, test_loader)

# Plot metrics
plot_metrics(train_loss, [accuracy]*NUM_EPOCHS)

# Print final evaluation results
print(f"Final Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, AUC ROC: {auc_roc:.4f}")
print("Confusion Matrix:\n", cm)

# End timing
end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")

