import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import matplotlib.pyplot as plt

class NormalizeTransform(object):
    def _call_(self, image):
        normalized_image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
        return normalized_image 

def assign_label(label):
    return 0 if not label else 1

class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = assign_label(target)
        return img, target

def load_dataset(dataset_name, training_size, image_size, seed=123):

    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder('./dataset/' + str(dataset_name) + '/train/', transform=transform)
    test_dataset = CustomImageFolder('./dataset/' + str(dataset_name) + '/test/', transform=transform)
    mask_dataset = CustomImageFolder('./dataset/' + str(dataset_name) + '/ground_truth/', transform=transform)

    subset_size_training = training_size 
    subset_indices_training = torch.randperm(len(train_dataset))[:subset_size_training]
    train_dataset = torch.utils.data.Subset(train_dataset, subset_indices_training)

    '''
    subset_size_test = len(test_dataset)
    subset_indices_test = torch.randperm(len(test_dataset))[:subset_size_test]
    test_dataset = torch.utils.data.Subset(test_dataset, subset_indices_test)
    '''
    
    batch_size = 4

    m=len(train_dataset)

    train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
    batch_size = 4

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    mask_loader = torch.utils.data.DataLoader(mask_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, mask_loader

if __name__ == "__main__":
    
    image_size = 32
    
    #train_loader, valid_loader, test_loader = load_malaria()
    train_loader, valid_loader, test_loader, mask_loader = load_dataset('carpet', image_size)

    plt.figure(figsize=(20,10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        img = train_loader.dataset[i][0].unsqueeze(0)
        ax.set_title(str(train_loader.dataset[i][1]), y=0, pad=-15)
        plt.axis("off")
        plt.imshow(img.squeeze().numpy(), cmap='gray')
        plt.savefig("./aaaa.png")
    plt.figure(figsize=(20,10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        img = test_loader.dataset[i][0].unsqueeze(0)
        ax.set_title(str(test_loader.dataset[i][1]), y=0, pad=-15)
        plt.axis("off")
        plt.imshow(img.squeeze().numpy(), cmap='gray')
        plt.savefig("./bbbb.png")
