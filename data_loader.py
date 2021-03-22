import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data(data_dir, mode):

    mfccs = np.load(os.path.join(data_dir, '{:s}_mfccs.npy'.format(mode)))
    annotations = np.load(os.path.join(data_dir, '{:s}_annotations.npy'.format(mode)))

    data, target = map(torch.tensor, (mfccs.astype(np.float32),
                                      annotations.astype(np.float32)))

    return data, target


def normalize_mfccs(sample_mfcc, mfcc_mean, mfcc_std):

    return (sample_mfcc - mfcc_mean) / mfcc_std


def make_training_loaders(data_dir):

    train_data, train_annotations = load_data(data_dir, 'train')
    test_data, test_annotations = load_data(data_dir, 'test')

    mfcc_mean, mfcc_std = torch.mean(train_data), torch.std(train_data)
    train_data = normalize_mfccs(train_data, mfcc_mean, mfcc_std)
    test_data = normalize_mfccs(test_data, mfcc_mean, mfcc_std)

    train_dataset = TensorDataset(train_data, train_annotations)
    test_dataset = TensorDataset(test_data, test_annotations)

    train_loader = DataLoader(train_dataset, batch_size=64, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, drop_last=True)

    return train_loader, test_loader


def make_testing_loader(data_dir):

    train_data, _ = load_data(data_dir, 'train')
    test_data, test_annotations = load_data(data_dir, 'test')

    mfcc_mean, mfcc_std = torch.mean(train_data), torch.std(train_data)
    test_data = normalize_mfccs(test_data, mfcc_mean, mfcc_std)

    test_dataset = TensorDataset(test_data, test_annotations)
    test_loader = DataLoader(test_dataset, batch_size=64, drop_last=True)

    return test_loader
