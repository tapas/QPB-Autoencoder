import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

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

def load_mnist_dataset(training_size, image_size):
    data_dir = 'dataset'

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
    test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)

    normal = 1
    anomaly = 7

    train_idx = train_dataset.targets == normal
    train_dataset.targets = train_dataset.targets[train_idx]
    train_dataset.data = train_dataset.data[train_idx]

    test_idx = (test_dataset.targets == anomaly) | (test_dataset.targets == normal)
    test_dataset.targets = test_dataset.targets[test_idx]
    test_dataset.targets = [0 if test_dataset.targets[i]==normal else 1 for i in range(len(test_dataset.targets))]
    test_dataset.data = test_dataset.data[test_idx]
    
    subset_size_training = training_size
    subset_indices_training = torch.randperm(len(train_dataset))[:subset_size_training]
    train_dataset = torch.utils.data.Subset(train_dataset, subset_indices_training)

    subset_size_test = 64
    subset_indices_test = torch.randperm(len(test_dataset))[:subset_size_test]
    test_dataset = torch.utils.data.Subset(test_dataset, subset_indices_test)
    

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    m=len(train_dataset)
    #m = subset_size_training

    train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
    batch_size = 4

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader, None

def load_malaria():

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder('./dataset/cell_images/training', transform=transform)
    valid_dataset = datasets.ImageFolder('./dataset/cell_images/validation', transform=transform)
    test_dataset = CustomImageFolder('./dataset/cell_images/test', transform=transform)

    subset_size_training = 1000
    subset_indices_training = torch.randperm(len(train_dataset))[:subset_size_training]
    train_dataset = torch.utils.data.Subset(train_dataset, subset_indices_training)

    subset_size_test = 200
    subset_indices_test = torch.randperm(len(test_dataset))[:subset_size_test]
    test_dataset = torch.utils.data.Subset(test_dataset, subset_indices_test)
    
    subset_size_validation = 200
    subset_indices_validation = torch.randperm(len(valid_dataset))[:subset_size_validation]
    valid_dataset = torch.utils.data.Subset(valid_dataset, subset_indices_validation)
    
    batch_size = 40

    m=len(train_dataset)
    #m = subset_size_training

    train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    '''
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
    '''

    return train_loader, valid_loader, test_loader

def load_MVTEC(dataset_name, training_size, image_size, seed=123):

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


def create_mask_normal(dataset_name, normal_images):
    img = np.zeros((256, 256))
    for i in range(28):
        if i < normal_images:
            s = "00"+str(i)+"_mask.png"
        else:
            s = "0"+str(i)+"_mask.png"
        black_image = Image.fromarray(img, mode="L")
        black_image.save("./dataset/" + str(dataset_name) + "/ground_truth/agood/" + s)

if __name__ == "__main__":
    
    image_size = 32
    
    
    #create_mask_normal("cartpet", 28)
    #exit()
    

    #train_loader, valid_loader, test_loader = load_malaria()
    train_loader, valid_loader, test_loader, mask_loader = load_MVTEC('carpet', image_size)

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