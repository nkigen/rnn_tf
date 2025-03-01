import os
import time

import torch
import utils
from models import CNNNetwork
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class_names = utils.class_names
# Device configuration
device = utils.set_accelerator()

# Hyperparameters
batch_size = 64
learning_rate = 0.001
weight_decay=1e-4
num_epochs = 20

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# Load FashionMNIST dataset
train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")

# Print class distribution
class_counts = [0] * 10
for _, label in train_dataset:
    class_counts[label] += 1

print("\nClass distribution in training set:")
for i, (name, count) in enumerate(zip(class_names, class_counts)):
    print(f"  {name}: {count} images")
    

# Initialize the model
model = CNNNetwork().to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=1, verbose=True
)

# Training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    
    total_images_processed = 0
    running_loss = 0.0
    correct = 0
    start_time = time.time()
    
    print(f"\nTraining on {size} images in {num_batches} batches of size {batch_size}:")
   
   
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        current_batch_size = X.shape[0]

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_images_processed += current_batch_size
        running_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss_value = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
            current = batch * batch_size + current_batch_size
            percent_complete = 100 * total_images_processed / size
            elapsed_time = time.time() - start_time
            images_per_second = total_images_processed / elapsed_time if elapsed_time > 0 else 0
            
            print(f"Batch {batch+1}/{num_batches} | "
                  f"Images: {total_images_processed}/{size} ({percent_complete:.1f}%) | "
                  f"Loss: {loss.item():.4f} | "
                  f"Speed: {images_per_second:.1f} img/s")
    
    epoch_loss = running_loss / num_batches
    epoch_accuracy = correct / size
    total_time = time.time() - start_time
    print(f"\nEpoch completed in {total_time:.2f}s | Avg Loss: {epoch_loss:.4f} | Training Accuracy: {(100*epoch_accuracy):.2f}%")
    print(f"Total images processed: {total_images_processed} at {total_images_processed/total_time:.1f} img/s")


# Testing function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

# Train the model
best_accuracy = 0
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    current_accuracy = test(test_loader, model, loss_fn)
    
    # Save the model if it's the best so far
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        
        # Create directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save just the model parameters
        torch.save(model.state_dict(),utils.model_param_file_name)
        
        print(f"Model saved with accuracy: {(100*current_accuracy):>0.1f}%")
    else:
        print(f"No improvement in accuracy ({(100*current_accuracy):>0.1f}% vs previous best {(100*best_accuracy):>0.1f}%)")


print("Training completed!")
