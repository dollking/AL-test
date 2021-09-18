import os
import random
from tqdm import tqdm

import numpy as np

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

    def sampling(self, step_cnt, strategy, task):
        if not step_cnt:
            random.shuffle(self.unlabeled)
            self.labeled = self.unlabeled[:self.initial_size]
            self.unlabeled = self.unlabeled[self.initial_size:]

            return

        sample_size = self.budget

        # unlabeled
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                pin_memory=self.config.pin_memory, sampler=Sampler(self.unlabeled))
        tqdm_batch = tqdm(dataloader, leave=False, total=len(dataloader))
        index = 0
        data_dict = {}
        for curr_it, data in enumerate(tqdm_batch):
            data = data[0].cuda(async=self.config.async_loading)

            _, features, _ = task.get_result(data)
            code = strategy.get_code(features)

            code = tuple(map(tuple, code.view([-1, self.config.vae_embedding_dim]).cpu().tolist()))

            for idx in range(len(code)):
                if code[idx] in data_dict:
                    data_dict[code[idx]].append(self.unlabeled[index])
                else:
                    data_dict[code[idx]] = [self.unlabeled[index]]
                index += 1
        tqdm_batch.close()

        for code in data_dict:
            random.shuffle(data_dict[code])

        # labeled
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                pin_memory=self.config.pin_memory, sampler=Sampler(self.labeled))
        tqdm_batch = tqdm(dataloader, leave=False, total=len(dataloader))
        loss_list = []
        for curr_it, data in enumerate(tqdm_batch):
            data = data[0].cuda(async=self.config.async_loading)

            _, features, pred_loss = task.get_result(data)
            code = strategy.get_code(features)

            code = tuple(map(tuple, code.view([-1, self.config.vae_embedding_dim]).cpu().tolist()))
            pred_loss = pred_loss.cpu().numpy()

            loss_list.extend([[code[idx], pred_loss[idx]] for idx in range(len(code))])
        tqdm_batch.close()

        loss_list.sort(key=lambda x: x[1], reverse=True)
        loss_list = np.array(loss_list[:2000])

        code_list = list(loss_list[:, 0])
        code_list.sort(key=lambda x: len(data_dict[x]) if x in data_dict else 0)

        # sampling
        index = 0
        sample_set = []
        while code_list and len(sample_set) < sample_size:
            tmp_code = code_list[index]
            if tmp_code in data_dict and data_dict[tmp_code]:
                sample_set.append(data_dict[tmp_code].pop())
                index += 1
            else:
                code_list.pop(index)

            index %= len(code_list)
        else:
            if len(sample_set) < sample_size:
                self.unlabeled = list(set(self.unlabeled) - set(sample_set))
                random.shuffle(self.unlabeled)
                sample_set += self.unlabeled[:sample_size - len(sample_set)]

        if len(set(sample_set)) < sample_size:
            print('!!!!!!!!!!!!!!!! error !!!!!!!!!!!!!!!!', len(set(sample_set)))

        self.labeled += sample_set
        self.unlabeled = list(set(self.unlabeled) - set(sample_set))
