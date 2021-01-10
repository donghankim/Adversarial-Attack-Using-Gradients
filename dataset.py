import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os, pdb
from PIL import Image


# class labels are from the imagenet database.
class DataSet():
    def __init__(self, config):
        self.root_dir = config.root_dir
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.train = None
        self.test = None
        self.train_loader = None
        self.test_loader = None
        self.classes = {}

        self.transforms = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        self.inv_norm = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

        # hardcode cifar10 labels
        if config.dataset == 'cifar10':
            self.classes[0] = 'airplane'
            self.classes[1] = 'automobile'
            self.classes[2] = 'bird'
            self.classes[3] = 'cat'
            self.classes[4] = 'deer'
            self.classes[5] = 'dog'
            self.classes[6] = 'frog'
            self.classes[7] = 'horse'
            self.classes[8] = 'ship'
            self.classes[9] = 'truck'

    def show_img(self, img, denormalize=True):
        if denormalize:
            img = self.inv_norm(img)

        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        plt.show()


# For cifar10 dataset
class Cifar10(DataSet):
    def __init__(self, config):
        super().__init__(config)
        self.create_dataset()

    def create_dataset(self):
        self.train = datasets.CIFAR10(self.root_dir, train = True, transform = self.transforms, download = True)
        self.test = datasets.CIFAR10(self.root_dir, train = False, transform = self.transforms, download = True)
        self.train_loader = DataLoader(self.train, batch_size = self.batch_size, shuffle = True)
        self.test_loader = DataLoader(self.test, batch_size = self.batch_size, shuffle = True)


# For Imagenet dataset (not complete)
class Imagenet(DataSet):
    def __init__(self, config):
        super().__init__(config)
        self.imagenet_dir = config.imagenet_dir
        self.create_dataset()

    def create_dataset(self):
        dataset = datasets.ImageFolder(self.imagenet_dir, transform = self.transforms)
        self.classes = dataset.classes
        self.test = torch.utils.data.Subset(dataset, range(self.test_size))
        self.train = torch.utils.data.Subset(dataset, range(self.test_size, len(dataset)))

        self.train_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test, batch_size=self.batch_size, shuffle=True)

# Delete before push
class Single_image(DataSet):
    def __init__(self, config):
        super().__init__(config)
        img = Image.open('bull_test.jpg')
        self.img = self.transforms(img)





