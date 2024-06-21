import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os


class ChromosomeDataset(Dataset):
    def __init__(self, tsv_dir, bin_size=10000):
        self.bin_size = bin_size
        self.processed_data = []

        for tsv_file in os.listdir(tsv_dir):
            data = pd.read_csv(tsv_dir + tsv_file, sep='\t', header=None, names=['chromosome', 'position', 'ignored'])
            processed_data = self.process_data(data)
            self.processed_data.append(processed_data)

        self.processed_data = torch.tensor(self.processed_data, dtype=torch.float32)

    def process_data(self, data):
        chromosome_vectors = []
        for chromosome in data['chromosome'].unique():
            chromosome_data = data[data['chromosome'] == chromosome]
            max_position = chromosome_data['position'].max()
            num_bins = max_position // self.bin_size + 1
            bins = np.zeros(num_bins, dtype=int)

            for _, row in chromosome_data.iterrows():
                bin_index = row['position'] // self.bin_size
                bins[bin_index] += 1

            chromosome_vectors.append(bins)

        return np.concatenate(chromosome_vectors)

    def __len__(self):
        return len(self.processed_data)  # We're treating the entire processed data as one sample

    def __getitem__(self, idx):
        return self.processed_data[idx]


def create_dataloader(tsv_dir, batch_size=1, bin_size=10000):
    dataset = ChromosomeDataset(tsv_dir, bin_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Usage example:
if __name__ == "__main__":
    tsv_dir = '/home/siweideng/OxTium_cfDNA'

    dataloader = create_dataloader(tsv_dir)

    for batch in dataloader:
        print(batch.shape)  # This will be a tensor of shape (1, N) where N is the total length of the concatenated vector
        # Your training or processing code here