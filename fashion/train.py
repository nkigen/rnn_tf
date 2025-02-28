# --- Training code part (train.py) ---
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Device configuration
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"using {device} device")

# Hyperparameters
batch_size = 64
learning_rate = 0.0001
num_epochs = 20

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load FashionMNIST dataset
train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
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
    
class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Initialize the model
model = NeuralNetwork().to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    
    total_images_processed = 0
    running_loss = 0.0
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
    total_time = time.time() - start_time
    print(f"\nEpoch completed in {total_time:.2f}s | Avg Loss: {epoch_loss:.4f}")
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
        
        # Option 1: Save the entire model
        torch.save(model, 'models/fashion_mnist_model.pth')
        
        # Option 2: Save just the model parameters (recommended)
        torch.save(model.state_dict(), 'models/fashion_mnist_model_params.pth')
        
        print(f"Model saved with accuracy: {(100*current_accuracy):>0.1f}%")
    else:
        print(f"No improvement in accuracy ({(100*current_accuracy):>0.1f}% vs previous best {(100*best_accuracy):>0.1f}%)")


print("Training completed!")
