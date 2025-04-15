import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt

torch.manual_seed(1)

ROOT_DIR = './Images/Retina-SLO/'
VALID_SPLIT = 0.15
TEST_SPLIT = 0.15
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

torch.manual_seed(1) # for reproducibility

class SLODataset(Dataset):
    def __init__(self, dataframe, IMAGE_SIZE, pretrained, root_dir=None, train=True):
        self.dataframe = dataframe
        self.IMAGE_SIZE = IMAGE_SIZE
        self.pretrained = pretrained
        self.train = train
        self.root_dir = root_dir
        if self.root_dir is None:
             raise ValueError("root_dir must be provided to SLODataset")

    def __len__(self):
        return len(self.dataframe)
    
    def get_transform(self, image):
        if self.train:
            # here can add augmentations to be used in training
            transform = transforms.Compose([
                transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),  
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # from https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.efficientnet_b0
            ])
        else: 
            # these are the transformations which will be used for inference only
            transform = transforms.Compose([
                            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                        ]) 
        return transform(image)

    def __getitem__(self, idx):
        """
        Get an instance from the dataset
        Input: id of image
        Returns:
            image
            label: tensor, contains label for ME and DR
        """
        relative_img_path = self.dataframe.iloc[idx]['img_path']
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx]['img_path'])
        image = Image.open(img_path)
        image = self.get_transform(image)
        label_me = int(self.dataframe.iloc[idx]['ME']) 
        label_dr = int(self.dataframe.iloc[idx]['DR'])
        label_glaucoma = int(self.dataframe.iloc[idx]['glaucoma'])

        return image, label_me, label_dr, label_glaucoma


def split_datasets(df, pretrained, root_dir):
    """
    Function to prepare and split the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along 
    with the class names.
    """
    dataset = SLODataset(df, IMAGE_SIZE, pretrained, root_dir=root_dir)

    dataset_size = len(dataset)
    
    # Calculate the validation + test dataset size.
    valid_size = int(VALID_SPLIT*dataset_size)
    test_size = int(TEST_SPLIT*dataset_size)
    train_size = dataset_size - valid_size - test_size
    
    # Radomize the data indices.
    indices = torch.randperm(len(dataset)).tolist()
    
    # Training and validation sets.
    dataset_train = Subset(dataset, indices[:train_size])
    dataset_valid = Subset(dataset, indices[train_size:(train_size + valid_size)])
    dataset_test = Subset(dataset, indices[(train_size+valid_size):])
    return dataset_train, dataset_valid, dataset_test

def get_data_loaders(dataset_train, dataset_valid, dataset_test, batch_size=BATCH_SIZE):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size, 
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        dataset_test, batch_size,
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader, test_loader


# testing the code
"""
df = pd.read_csv("/Users/oskar/Documents/Research/Miguel + Sam/Code/Images/Retina-SLO_dataset/Retina-SLO_labels/Retina-SLO.txt",
                  sep=":|,", engine='python')
df = df[:16]
df.rename(columns={'img_path ': 'img_path', ' ME_GT': 'ME', ' DR_GT': 'DR', ' glaucoma_GT': 'glaucoma'}, inplace=True)
df['glaucoma'] = df['glaucoma'].str.strip("; ")
df['img_path'] = df['img_path'].str.strip()
df = df[df['img_path'].str.contains('study1')]
df.reset_index(drop=True, inplace=True)

image_path_start = '/Users/oskar/Documents/Research/Miguel + Sam/Code/Images/Retina-SLO_dataset/'

# path = image_path_start + img_path

train_set, val_set, test_set = split_datasets(pretrained=True)

train_loader, val_loader, test_loader = get_data_loaders(train_set, val_set, test_set)

train_features, train_labels = next(iter(train_loader))
train_features, train_labels = next(iter(train_loader))
train_features, train_labels = next(iter(train_loader))
print("last batch")
train_features, train_labels = next(iter(train_loader))

img = train_features[0].squeeze().T
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")"""
