import os
from PIL import Image
import numpy as np
import torch
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split, SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder

class FoodDataset(Dataset):
    def __init__(self, data_dir, split_file, classes=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = classes

        with open(split_file, 'r') as f:
            self.image_files = f.read().splitlines()

        self.images, self.labels = self.load_data()
        
        # create and fit the LabelEncoder
        self.le = LabelEncoder()
        self.le.fit(self.labels)
        self.labels = self.le.transform(self.labels)

        with open('../models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.le, f)

    def load_data(self):
        images = []
        labels = []
        for image_file in self.image_files:
            label = image_file.split('/')[0]
            if self.classes is None or label in self.classes:
                image_path = os.path.join(self.data_dir, image_file + '.jpg')
                with Image.open(image_path).convert('RGB') as img:
                    images.append(img.copy())
                labels.append(label)
        return images, labels


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label).long()
        return image, label

def prep_data(data_dir, split_file, classes, batch_size, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        # maybe add random rotations here
        # Image Net dataset normalization values 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FoodDataset(data_dir, split_file, classes, transform)

    # split train data into (train, validation) only if train.txt is given
    if 'train' in split_file:
        val_size = int(val_split * len(dataset))
        # train_size = len(dataset) - val_size

        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        return train_loader, val_loader
    
    # else this is test.txt
    else:
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return test_loader
