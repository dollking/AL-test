import os
import random
from tqdm import tqdm

import numpy as np

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10

from data.sampler import Sampler

cudnn.benchmark = False


class Query(object):
    def __init__(self, config):
        self.config = config

        self.initial_size = self.config.initial_size
        self.budget = self.config.budge_size
        self.labeled = []
        self.unlabeled = [i for i in range(self.config.data_size)]

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])

        self.batch_size = self.config.batch_size

        # define dataloader
        if self.config.data_name == 'cifar10':
            self.dataset = CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                   train=True, download=True, transform=self.train_transform)
        elif self.config.data_name == 'cifar100':
            self.dataset = CIFAR100(os.path.join(self.config.root_path, self.config.data_directory),
                                    train=True, download=True, transform=self.train_transform)

    def sampling(self, step_cnt, task):
        sample_size = self.budget if step_cnt else self.initial_size
        random.shuffle(self.unlabeled)

        if step_cnt:
            subset = self.unlabeled[:sample_size * 10]

            dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                    pin_memory=self.config.pin_memory, sampler=Sampler(subset))

            uncertainty = torch.tensor([]).cuda()
            tqdm_batch = tqdm(dataloader, total=len(dataloader))
            for curr_it, data in enumerate(tqdm_batch):
                data = data[0].cuda(async=self.config.async_loading)

                _, _, pred_loss = task.get_result(data)

                uncertainty = torch.cat([uncertainty, pred_loss], 0)
            tqdm_batch.close()

            uncertainty = uncertainty.cpu()
            arg = np.argsort(uncertainty)

            sample_set = list(torch.tensor(subset)[arg][-sample_size:].numpy())

        else:
            sample_set = self.unlabeled[:sample_size]

        self.labeled += sample_set
        self.unlabeled = list(set(self.unlabeled) - set(sample_set))
