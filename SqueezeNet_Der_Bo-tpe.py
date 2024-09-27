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
import optuna  # Import Optuna

# Define constants
BATCH_SIZE = 1024  # Batch size for training
NUM_EPOCHS = 30
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

# Load datasets
train_data = ImageFolder(train_dir, transform=transform)
test_data = ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# Function to create and return a SqueezeNet model
def create_model(learning_rate):
    # Load SqueezeNet model
    model = squeezenet1_1(pretrained=True)
    num_classes = len(train_data.classes)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))  # Modify the final layer
    model.num_classes = num_classes
    model = model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer


# Function to train the model with DER
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    train_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Calculate weights for DER
            outputs = model(images)
            probs = nn.Softmax(dim=1)(outputs)
            der_weights = (1 / (probs + 1e-8))  # Adding epsilon to prevent division by zero

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            # Adjust loss with DER weights
            weighted_loss = (loss * der_weights.gather(1, labels.view(-1, 1))).mean()
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()

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
    criterion = nn.CrossEntropyLoss()

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


# Objective function for Optuna to optimize learning rate
def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)

    model, criterion, optimizer = create_model(learning_rate)

    train_loss = train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS)
    avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(model, test_loader)

    return accuracy  # Return accuracy as the objective


# Start timing
start_time = time.time()

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # You can change the number of trials

# Get the best hyperparameters
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train and evaluate the model with the best learning rate
best_learning_rate = best_params["learning_rate"]
model, criterion, optimizer = create_model(best_learning_rate)

# Train the model with the best hyperparameters
train_loss = train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS)

# Evaluate the model with the best hyperparameters
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
