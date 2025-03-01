import torch


def set_accelerator():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"using {device} device")
    return device
    
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model_dir="models"
model_file_suffix='model.pth'
model_param_file_suffix='model_params.pth'


# def load_dataset(is_train=False):
    