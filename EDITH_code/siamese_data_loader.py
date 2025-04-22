import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class DynamicSiameseDataset(Dataset):
    def __init__(self, data, labels, negative_prob=0.5):
        """
        Initialize the dataset with the full data and labels.

        Args:
            data (np.array): The feature data.
            labels (np.array): The labels corresponding to the data.
            negative_prob (float): Probability of choosing a negative pair.
        """
        self.data = data
        self.labels = labels
        self.negative_prob = negative_prob
        self.label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}

    def __getitem__(self, index):
        """
        Generate one sample of the dataset - a pair of data points (positive or negative).

        Args:
            index (int): Index of the first data point.

        Returns:
            tuple: A sample pair with labels (1 for positive, 0 for negative).
        """
        label1 = self.labels[index]
        data1 = self.data[index]
        
        # Decide whether to return a positive or negative example
        if random.random() > self.negative_prob:
            # Positive pair
            positive_indices = self.label_to_indices[label1]
            index2 = index
            while index2 == index:
                index2 = np.random.choice(positive_indices)
            label = 1
        else:
            # Negative pair
            negative_labels = list(self.label_to_indices.keys())
            negative_labels.remove(label1)
            label2 = np.random.choice(negative_labels)
            index2 = np.random.choice(self.label_to_indices[label2])
            label = 0
        
        data2 = self.data[index2]

        return data1, data2, label

    def __len__(self):
        return len(self.data)

