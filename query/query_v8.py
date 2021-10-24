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
            indices = tuple(map(tuple, indices.cpu().tolist()))

            for idx in range(len(indices)):
                index_lst += list(set(map(tuple, indices[idx])))

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

        data_lst = []
        for curr_it, data in enumerate(tqdm_batch):
            inputs = data[0].cuda(async=self.config.async_loading)
            targets = data[1].cuda(async=self.config.async_loading)

            _, features, loss = task.get_result(inputs, targets)
            loss = loss.cpu().numpy()

            indices = strategy.get_index(inputs)
            indices = tuple(map(tuple, indices.cpu().tolist()))

            for idx in range(len(indices)):
                data_lst.append([loss[idx], indices[idx]])
        tqdm_batch.close()

        data_lst = sorted(data_lst, key=lambda x: x[0], reverse=True)

        #############################
        data_index_set = []
        for data in data_lst:
            data_index_set.extend(list(data[1]))
        data_index_set = set(map(tuple, data_index_set))

        #############################
        labeled_index_set = []
        for data in data_lst[:int(self.initial_size * 0.6)]:
            labeled_index_set.extend(list(data[1]))
        labeled_index_set = list(map(tuple, labeled_index_set))

        labeled_index_cnt = Counter(labeled_index_set)

        #############################
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                pin_memory=self.config.pin_memory, sampler=Sampler(self.unlabeled))
        tqdm_batch = tqdm(dataloader, leave=False, total=len(dataloader))

        index = 0
        unlabeled_set = []
        for curr_it, data in enumerate(tqdm_batch):
            inputs = data[0].cuda(async=self.config.async_loading)

            code = strategy.get_code(inputs)
            code = code.view([-1, self.config.vae_embedding_dim, code.size(2) * code.size(3)]).transpose(1, 2)
            code = tuple(map(tuple, code.cpu().tolist()))

            for idx in range(len(code)):
                tmp_code = tuple(map(tuple, code[idx]))
                if use_labeled_cnt:
                    unlabeled_set.append([self.unlabeled[index],
                                          sum([labeled_index_cnt[key] * self.index_idf[key] for key in
                                               set(tmp_code) & data_index_set])])
                else:
                    unlabeled_set.append([self.unlabeled[index],
                                          sum([self.index_idf[key] for key in set(tmp_code) & data_index_set])])
                index += 1
        tqdm_batch.close()

        sample_set = list(np.array(sorted(unlabeled_set, key=lambda x: x[1]))[:sample_size, 0])

        if len(set(sample_set)) < sample_size:
            print('!!!!!!!!!!!!!!!! error !!!!!!!!!!!!!!!!', len(set(sample_set)))

        self.labeled += sample_set
        self.unlabeled = list(set(self.unlabeled) - set(sample_set))
