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

    def sampling(self, step_cnt, task, ae):
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

            ##########################################################################
            ##########################################################################

            dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                    pin_memory=self.config.pin_memory, sampler=Sampler(self.labeled))
            labeled_features = torch.tensor([]).cuda()
            tqdm_batch = tqdm(dataloader, total=len(dataloader))
            for curr_it, data in enumerate(tqdm_batch):
                data = data[0].cuda(async=self.config.async_loading)

                ae_features = ae.get_feature(data)
                ae_features = ae_features.view([-1, self.config.vae_embedding_dim])

                labeled_features = torch.cat([labeled_features, ae_features], 0)
            tqdm_batch.close()
            ##############################################################################
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                    pin_memory=self.config.pin_memory, sampler=Sampler(self.unlabeled))
            uncertainty = torch.tensor([]).cuda()
            feature_distances = torch.tensor([]).cuda()
            tqdm_batch = tqdm(dataloader, total=len(dataloader))
            for curr_it, data in enumerate(tqdm_batch):
                targets = data[1].cuda(async=self.config.async_loading)
                data = data[0].cuda(async=self.config.async_loading)

                _, _, loss = task.get_result2(data, targets)
                uncertainty = torch.cat([uncertainty, loss], 0)

                ae_features = ae.get_feature(data)
                ae_features = ae_features.view([-1, self.config.vae_embedding_dim])

                distances = (torch.sum(ae_features ** 2, dim=1, keepdim=True) + torch.sum(labeled_features ** 2, dim=1)
                             - 2 * torch.matmul(ae_features, labeled_features.t()))
                distances = torch.sum(distances, dim=1)

                feature_distances = torch.cat([feature_distances, distances], 0)

            tqdm_batch.close()

            uncertainty = uncertainty.cpu()
            arg = np.argsort(uncertainty)

            feature_distances = feature_distances.cpu()
            distance_lst = feature_distances[arg].numpy()
            high_lst, low_lst = distance_lst[:len(distance_lst) // 4], distance_lst[-(len(distance_lst) // 4):]
            print(high_lst.mean(), high_lst.std())
            print(low_lst.mean(), low_lst.std())


        else:
            sample_set = self.unlabeled[:sample_size]

        self.labeled += sample_set
        self.unlabeled = list(set(self.unlabeled) - set(sample_set))
