import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

color_jitter_transform = transforms.ColorJitter(
    brightness=0.5,  # example range; adjust as needed
    contrast=0.5,   # example range; adjust as needed
    saturation=0.5, # example range; adjust as needed
    hue=0.1         # example range; adjust as needed
)

transform = transforms.Compose([
    transforms.Resize((224,224)),  # Resize to 224x224
    transforms.RandomApply([color_jitter_transform], p=0.5),
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.3),
    transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2))], p=0.3),
    transforms.ToTensor(),   # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

train_dataset = datasets.ImageFolder(root='/Users/aishwaryamallikarjun/Desktop/Intro to Deep Learning/archive/train', transform=transform)
test_dataset = datasets.ImageFolder(root='/Users/aishwaryamallikarjun/Desktop/Intro to Deep Learning/archive/test', transform=transform)
valid_dataset = datasets.ImageFolder(root='/Users/aishwaryamallikarjun/Desktop/Intro to Deep Learning/archive/valid', transform=transform)

batch_size = 16

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def get_train_loader():
    return train_loader

def get_test_loader():
    return test_loader

def get_val_loader():
    return val_loader