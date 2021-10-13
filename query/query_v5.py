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
        self.code_idf = {}

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

        code_lst = []
        for curr_it, data in enumerate(tqdm_batch):
            inputs = data[0].cuda(async=self.config.async_loading)

            code = strategy.get_code(inputs)
            code = code.view([-1, self.config.vae_embedding_dim, code.size(2) * code.size(3)]).transpose(1, 2)
            code = tuple(map(tuple, code.cpu().tolist()))

            for idx in range(len(code)):
                code_lst += list(set(map(tuple, code[idx])))

        code_cnt = Counter(code_lst)

        for key in code_cnt:
            self.code_idf[key] = math.log(self.config.data_size / (1 + code_cnt[key]))

    def sampling(self, step_cnt, strategy, task):
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

            code = strategy.get_code(inputs)
            code = code.view([-1, self.config.vae_embedding_dim, code.size(2) * code.size(3)]).transpose(1, 2)
            code = tuple(map(tuple, code.cpu().tolist()))

            for idx in range(len(code)):
                data_lst.append([loss[idx], code[idx]])
        tqdm_batch.close()

        data_lst = sorted(data_lst, key=lambda x: x[0], reverse=True)

        ############################# diversity
        diversity_feature_set = []
        for data in data_lst:
            diversity_feature_set.extend(list(data[1]))
        diversity_feature_set = set(map(tuple, diversity_feature_set))

        ############################# uncertainty
        uncertainty_feature_set = []
        for data in data_lst[:int(self.initial_size * 0.6)]:
            uncertainty_feature_set.extend(list(data[1]))
        uncertainty_feature_set = list(map(tuple, uncertainty_feature_set))

        feature_cnt = Counter(uncertainty_feature_set)
        for key in feature_cnt:
            feature_cnt[key] *= self.code_idf[key]
        feature_cnt = sorted(list(zip(feature_cnt.keys(), feature_cnt.values())), key=lambda x: x[1], reverse=True)

        uncertainty_feature_set, _ = zip(*feature_cnt[:len(feature_cnt) // 4])
        uncertainty_feature_set = set(uncertainty_feature_set)

        #############################
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                pin_memory=self.config.pin_memory, sampler=Sampler(self.unlabeled))
        tqdm_batch = tqdm(dataloader, leave=False, total=len(dataloader))

        index = 0
        unlabeled_set1, unlabeled_set2 = [], []
        for curr_it, data in enumerate(tqdm_batch):
            inputs = data[0].cuda(async=self.config.async_loading)

            code = strategy.get_code(inputs)
            code = code.view([-1, self.config.vae_embedding_dim, code.size(2) * code.size(3)]).transpose(1, 2)
            code = tuple(map(tuple, code.cpu().tolist()))

            for idx in range(len(code)):
                tmp_code = tuple(map(tuple, code[idx]))
                unlabeled_set1.append([self.unlabeled[index], len(set(tmp_code) & uncertainty_feature_set)])
                unlabeled_set2.append([self.unlabeled[index],
                                       sum([1 / self.code_idf[key] for key in set(tmp_code) & diversity_feature_set])])
                index += 1
        tqdm_batch.close()

        uncertainty_cnt = sample_size // 2
        sample_set = list(np.array(sorted(unlabeled_set1, key=lambda x: x[1], reverse=True))[:uncertainty_cnt, 0])

        idx = 0
        ordered_diversity_index = list(
            np.array(sorted(unlabeled_set2, key=lambda x: x[1]))[:sample_size - uncertainty_cnt, 0]
        )
        while len(sample_set) < sample_size:
            if ordered_diversity_index[idx] not in sample_set:
                sample_set.append(ordered_diversity_index[idx])
            idx += 1

        if len(set(sample_set)) < sample_size:
            print('!!!!!!!!!!!!!!!! error !!!!!!!!!!!!!!!!', len(set(sample_set)))

        self.labeled += sample_set
        self.unlabeled = list(set(self.unlabeled) - set(sample_set))
