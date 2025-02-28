import inference
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Example usage of inference
if __name__ == "__main__":
    # Device configuration
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"using {device} device")
    
    # Load the model (Option 2 - recommended)
    model = inference.load_model('models/fashion_mnist_model_params.pth', device)
    
    # Load a test image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )
    
    # Get a sample image
    img, label = test_dataset[3]
    
    # Make prediction
    predicted_idx, predicted_class = inference.predict_image(img, model, device)
    
    print(f"Predicted: {predicted_class}")
    print(f"Actual: {inference.class_names[label]}")