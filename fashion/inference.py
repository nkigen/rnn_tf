

# --- Inference code part (inference.py) ---
import torch
from torch import nn
from torchvision import datasets, transforms

# Class names for FashionMNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define the model architecture (must match the saved model)
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

def load_model(model_path, device='cpu'):
    # Option 1: Load the entire model
    if model_path.endswith('model.pth'):
        model = torch.load(model_path, map_location=device)
    
    # Option 2: Load just the parameters into the model architecture
    elif model_path.endswith('model_params.pth'):
        model = NeuralNetwork().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()  # Set to evaluation mode
    return model

def predict_image(img, model, device):
    # Add batch dimension and move to device
    x = img.unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        pred = model(x)
        predicted_idx = pred.argmax(1).item()
    
    return predicted_idx, class_names[predicted_idx]