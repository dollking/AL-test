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
        sample_size = self.budget if step_cnt else self.initial_size

        dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                pin_memory=self.config.pin_memory, sampler=Sampler(self.unlabeled))
        tqdm_batch = tqdm(dataloader, total=len(dataloader))

        index = 0
        data_dict = {}
        for curr_it, data in enumerate(tqdm_batch):
            data = data[0].cuda(async=self.config.async_loading)

            code = strategy.get_code(data)
            _, pred_loss = task.get_result(data)

            code = tuple(map(tuple, code.view([-1, self.config.vae_embedding_dim]).cpu().tolist()))
            pred_loss = pred_loss.cpu().numpy()

            for idx in range(len(code)):
                if code[idx] in data_dict:
                    data_dict[code[idx]].append([pred_loss[idx], self.unlabeled[index]])
                else:
                    data_dict[code[idx]] = [[pred_loss[idx], self.unlabeled[index]]]
                index += 1

        tqdm_batch.close()

        for i in data_dict:
            print(len(data_dict[i]), end=' / ')
        print()

        sample_set = []
        total_remain = []
        quota = int(sample_size / len(data_dict))
        for i in data_dict:
            if step_cnt:
                tmp_list = sorted(data_dict[i], key=lambda x: x[0], reverse=True)

                if len(tmp_list) > quota:
                    sample_set += list(np.array(tmp_list)[:quota, 1])
                    total_remain += tmp_list[quota:]
                else:
                    sample_set += list(np.array(tmp_list)[:, 1])

            else:
                tmp_list = set(np.array(data_dict[i])[:, 1])

                if len(tmp_list) > quota:
                    sampled = random.sample(tmp_list, quota)
                    sample_set += sampled
                    total_remain += list(tmp_list - set(sampled))
                else:
                    sample_set += list(tmp_list)

        if step_cnt:
            tmp_list = sorted(total_remain, key=lambda x: x[0], reverse=True)
            sample_set += list(np.array(tmp_list)[:(sample_size - len(sample_set)), 1])

        else:
            sample_set += random.sample(total_remain, sample_size - len(sample_set))

        print('cent cnt:', len(set(data_dict.keys())))

        if len(set(sample_set)) < sample_size:
            print('!!!!!!!!!!!!!!!! error !!!!!!!!!!!!!!!!', len(set(sample_set)))

        self.labeled += sample_set
        self.unlabeled = list(set(self.unlabeled) - set(sample_set))
