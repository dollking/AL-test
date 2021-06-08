import os
import numpy as np

import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, config, transform, data_list):
        self.root_path = config.root_path
        self.data_directory = config.data_directory
        self.config = config
        self.transform = transform
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = os.path.join(self.root_path, self.config.data_directory,
                                 self.config.data_name, self.data_list[idx].strip() + '.npz')
        data = np.load(file_path)

        return {'X': data['img'], 'target': data['label']}