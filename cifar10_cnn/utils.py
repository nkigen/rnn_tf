import torch


def set_accelerator():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"using {device} device")
    return device
    
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model_dir="models"
model_file_prefix="cifar10"
model_file_suffix='model.pth'
model_param_file_suffix='model_params.pth'
model_param_file_name=f'{model_dir}/{model_file_prefix}_{model_param_file_suffix}'
    