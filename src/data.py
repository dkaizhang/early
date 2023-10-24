import numpy as np
import pandas as pd
import random

from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms as T

def choose_split(split, train_data, val_data, test_data):
    if split == 'train':
        return train_data
    elif split == 'val':
        return val_data
    elif split == 'test':
        return test_data
    else:
        print("Invalid split, exiting...")
        exit(1)    

def sample_idx(len, frac, seed):

    random.seed(seed)
    frac_size = int(len * frac)
    return random.sample(range(len), frac_size)

class CSVDataset(Dataset):
    def __init__(self, data_csv, transform, convert=True):
        super().__init__()
        self.data_csv = pd.read_csv(data_csv)
        self.transform = transform
        self.convert = convert

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        image_file = self.data_csv.iloc[idx]['img_path']
        with Image.open(image_file) as img:

            if self.convert:
                img = img.convert('RGB')

            image = self.transform(img)
        label = int(self.data_csv.iloc[idx]['label'])
        
        return image, label


class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], int(self.labels[idx])
        image = Image.fromarray(image, mode='L')
        image = self.transform(image)
            
        return image, label

def get_transform(data):

    if data == 'MNIST':
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    (0.1307,), (0.3081,))
            ]
        ) 
    elif data == 'pneu':
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    (0.5711,),(0.1523,) # hand calc
                )
            ]
        )    
    else: 
        print("Data choice invalid, exiting...")
        exit(1)

    return transform

def load_pneu(decoy, split, seed, frac, convert=True):

    if frac is None:
        frac = 1.0

    transform = get_transform('pneu')

    if decoy == None:
        train_data = CSVDataset(data_csv='data/pneu/chest_xray/train/train_data.csv',transform=transform, convert=convert)
        val_data = CSVDataset(data_csv='data/pneu/chest_xray/val/val_data.csv',transform=transform, convert=convert)
        test_data = CSVDataset(data_csv='data/pneu/chest_xray/test/test_data.csv',transform=transform, convert=convert)
    elif decoy == 'text':
        train_data = CSVDataset(data_csv='data/pneu/chest_xray/train_text/train_data.csv',transform=transform, convert=convert)
        val_data = CSVDataset(data_csv='data/pneu/chest_xray/val_text/val_data.csv',transform=transform, convert=convert)
        test_data = CSVDataset(data_csv='data/pneu/chest_xray/test_text/test_data.csv',transform=transform, convert=convert)
    elif decoy == 'stripe':
        train_data = CSVDataset(data_csv='data/pneu/chest_xray/train_stripe/train_data.csv',transform=transform, convert=convert)
        val_data = CSVDataset(data_csv='data/pneu/chest_xray/val_stripe/val_data.csv',transform=transform, convert=convert)
        test_data = CSVDataset(data_csv='data/pneu/chest_xray/test_stripe/test_data.csv',transform=transform, convert=convert)
    else:
        print('not implemented')
        exit(0)
    
    if frac < 1.0:
        train_data = Subset(train_data, sample_idx(len(train_data), frac, seed))
        val_data = Subset(val_data, sample_idx(len(val_data), frac, seed))
        test_data = Subset(test_data, sample_idx(len(test_data), frac, seed))

    return choose_split(split, train_data, val_data, test_data)

def load_test_cor(data):

    if data == 'pneu_text_cor':
        transform = get_transform('pneu')
        test_data = CSVDataset(data_csv='data/pneu/chest_xray/test_text_cor/test_cor_data.csv',transform=transform, convert=True)
    elif data == 'decoyMNIST_cor':
        data = np.load('data/MNIST/decoyed_mnist.npz')
        transform = get_transform('MNIST')
        test_data = MNISTDataset(data['test_cor_images'], data['test_cor_labels'], transform=transform)
    return test_data

def load_decoyMNIST(split, seed, frac):

    if frac is None:
        frac = 1.0

    path = 'data/MNIST/decoyed_mnist.npz'
    data = np.load(path)

    transform = get_transform('MNIST')

    train_data = MNISTDataset(data['train_images'], data['train_labels'], transform=transform)
    val_data = MNISTDataset(data['val_images'], data['val_labels'], transform=transform)
    test_data = MNISTDataset(data['test_images'], data['test_labels'], transform=transform)
    if frac < 1.0:
        train_data = Subset(train_data, sample_idx(len(train_data), frac, seed))
        val_data = Subset(val_data, sample_idx(len(val_data), frac, seed))
        test_data = Subset(test_data, sample_idx(len(test_data), frac, seed))

    return choose_split(split, train_data, val_data, test_data)

def load_MNIST(split, seed, frac):
    
    if frac is None:
        frac = 1.0

    path = 'data/MNIST/original_mnist.npz'
    data = np.load(path)

    transform = get_transform('MNIST')

    train_data = MNISTDataset(data['train_images'], data['train_labels'], transform=transform)
    val_data = MNISTDataset(data['val_images'], data['val_labels'], transform=transform)
    test_data = MNISTDataset(data['test_images'], data['test_labels'], transform=transform)
    if frac < 1.0:
        train_data = Subset(train_data, sample_idx(len(train_data), frac, seed))
        val_data = Subset(val_data, sample_idx(len(val_data), frac, seed))
        test_data = Subset(test_data, sample_idx(len(test_data), frac, seed))
    
    return choose_split(split, train_data, val_data, test_data)

def load_data(data, split, seed, frac):

    if seed is None:
        seed = random.randint(0,1000)

    if data == 'decoyMNIST':
        return load_decoyMNIST(split=split, seed=seed, frac=frac) 
    elif data == 'decoyMNIST_cor':
        return load_test_cor(data=data)

    elif data == 'MNIST':
        return load_MNIST(split=split, seed=seed, frac=frac)

    elif data == 'pneu':
        return load_pneu(decoy=None, split=split, seed=seed, frac=frac)

    elif data == 'pneu_text':
        return load_pneu(decoy='text', split=split, seed=seed, frac=frac)
    elif data == 'pneu_text_cor':
        return load_test_cor(data=data)

    elif data == 'pneu_stripe':
        return load_pneu(decoy='stripe', split=split, seed=seed, frac=frac)
    
    else:
        print("Unsupported data, exiting...")
        exit(1) 

def load_labels(data):
    mnist_labels = [i for i in range(10)]
    pneu_labels = [0,1]

    if data[:10] == 'decoyMNIST' or data == 'MNIST':
        return mnist_labels
    elif data[0:4] == 'pneu':
        return pneu_labels
    else:
        print("Unsupported data, exiting...")
        exit(1) 