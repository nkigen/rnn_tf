import torch
import utils
from models import CNNNetwork

# Class names for FashionMNIST
class_names = utils.class_names


def load_model(model_path, device='cpu'):
    # Option 1: Load the entire model
    if model_path.endswith(utils.model_param_file_suffix):
        model = CNNNetwork().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    
        model.eval()  # Set to evaluation mode
    else:
        raise ValueError("Invalid model path")
    return model

def predict_image(img, model, device):
    # Add batch dimension and move to device
    x = img.unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        pred = model(x)
        print(pred)
        predicted_idx = pred.argmax(1).item()
    
    return predicted_idx, class_names[predicted_idx]