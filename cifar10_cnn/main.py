import inference
import utils
from torchvision import datasets, transforms

# Example usage of inference
if __name__ == "__main__":
    # Device configuration
    device = utils.set_accelerator()
    
    # Load the model (Option 2 - recommended)
    model_path=utils.model_param_file_name
    
    print(f"using path {model_path}")
    model = inference.load_model(model_path, device)
    
    # Load a test image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    test_dataset = datasets.CIFAR10(
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