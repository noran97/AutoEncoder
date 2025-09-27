import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def get_data(batch_size=128):
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download datasets
    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def visualize_batch(dataloader, classes=None):
    """عرض بعض الصور من batch"""
    images, labels = next(iter(dataloader))
    
    plt.figure(figsize=(10, 4))
    for i in range(8):  # عرض أول 8 صور
        plt.subplot(2, 4, i+1)
        # الصور متطبعة بين -1 و 1 → نرجعها لمجال [0,1] للعرض
        img = (images[i] * 0.5) + 0.5  
        plt.imshow(img.squeeze(), cmap="gray")
        
        if classes is not None:
            plt.title(classes[labels[i]])
        else:
            plt.title(f"Label: {labels[i].item()}")
        
        plt.axis("off")
    plt.show()
