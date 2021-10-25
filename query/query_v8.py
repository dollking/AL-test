import os
import math
import random
from tqdm import tqdm

import numpy as np
from collections import Counter

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10

from data.sampler import Sampler


class Query(object):
    def __init__(self, config):
        self.config = config

        self.initial_size = self.config.initial_size
        self.budget = self.config.budge_size
        self.labeled = []
        self.unlabeled = [i for i in range(self.config.data_size)]
        self.index_idf = {}

        self.batch_size = self.config.vae_batch_size

        # define dataloader
        if 'cifar' in self.config.data_name:
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ])

            if self.config.data_name == 'cifar10':
                self.dataset = CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                       train=True, download=True, transform=self.train_transform)
            elif self.config.data_name == 'cifar100':
                self.dataset = CIFAR100(os.path.join(self.config.root_path, self.config.data_directory),
                                        train=True, download=True, transform=self.train_transform)

    def set_idf(self, strategy):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                pin_memory=self.config.pin_memory, sampler=Sampler(self.unlabeled))
        tqdm_batch = tqdm(dataloader, leave=False, total=len(dataloader))

        index_lst = []
        for curr_it, data in enumerate(tqdm_batch):
            inputs = data[0].cuda(async=self.config.async_loading)

            indices = strategy.get_index(inputs)
            indices = indices.cpu().tolist()

            for idx in range(len(indices)):
                index_lst += indices[idx]

        index_cnt = Counter(index_lst)

        for key in index_cnt:
            self.index_idf[key] = math.log(self.config.data_size / (1 + index_cnt[key]))

    def sampling(self, step_cnt, strategy, task, use_labeled_cnt=False):
        if not step_cnt:
            random.shuffle(self.unlabeled)
            self.labeled = self.unlabeled[:self.initial_size]
            self.unlabeled = self.unlabeled[self.initial_size:]

            return

        sample_size = self.budget

        dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                pin_memory=self.config.pin_memory, sampler=Sampler(self.labeled))
        tqdm_batch = tqdm(dataloader, leave=False, total=len(dataloader))

        labeled_indices = []
        for curr_it, data in enumerate(tqdm_batch):
            inputs = data[0].cuda(async=self.config.async_loading)

            indices = strategy.get_index(inputs)
            indices = indices.cpu().tolist()

            for idx in range(len(indices)):
                labeled_indices += indices[idx]
        tqdm_batch.close()

        #############################
        labeled_index_cnt = Counter(labeled_indices)
        labeled_index_set = set(labeled_index_cnt.keys())


        #############################
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                pin_memory=self.config.pin_memory, sampler=Sampler(self.unlabeled))
        tqdm_batch = tqdm(dataloader, leave=False, total=len(dataloader))

        index = 0
        unlabeled_set = []
        for curr_it, data in enumerate(tqdm_batch):
            inputs = data[0].cuda(async=self.config.async_loading)

            indices = strategy.get_index(inputs)
            indices = indices.cpu().tolist()

            for idx in range(len(indices)):
                tmp_indices = set(indices[idx])
                if use_labeled_cnt:
                    unlabeled_set.append([self.unlabeled[index],
                                          sum([labeled_index_cnt[key] * self.index_idf[key] for key in
                                               set(tmp_indices) & labeled_index_set])])
                else:
                    unlabeled_set.append([self.unlabeled[index],
                                          sum([self.index_idf[key] for key in set(tmp_indices) & labeled_index_set])])
                index += 1
        tqdm_batch.close()

        sample_set = list(np.array(sorted(unlabeled_set, key=lambda x: x[1]))[:sample_size, 0])

        if len(set(sample_set)) < sample_size:
            print('!!!!!!!!!!!!!!!! error !!!!!!!!!!!!!!!!', len(set(sample_set)))

        self.labeled += sample_set
        self.unlabeled = list(set(self.unlabeled) - set(sample_set))
