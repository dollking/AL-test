import os
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10

from .graph.vae import VAE as vae
from data.sampler import Sampler


cudnn.benchmark = True


class Query(object):
    def __init__(self, config, data_size):
        self.config = config

        self.budget = self.config.budge_size
        self.labeled = []
        self.unlabeled = [i for i in range(data_size)]

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.batch_size = self.config.vae_batch_size

        # define dataloader
        self.cifar10_train = CIFAR10('data/cifar10', train=True, download=True, transform=self.train_transform)

        # define models
        self.vae = vae(self.config.vae_num_hiddens, self.config.vae_num_residual_layers,
                       self.config.vae_num_residual_hiddens, self.config.vae_num_embeddings,
                       self.config.vae_embedding_dim, self.config.vae_commitment_cost, self.config.vae_distance,
                       self.config.vae_decay).cuda()

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.vae = nn.DataParallel(self.vae, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint()

    def load_checkpoint(self):
        filename = os.path.join(self.config.root_path, self.config.checkpoint_directory, 'vae.pth.tar')

        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.vae.load_state_dict(checkpoint['vae_state_dict'])

            return True

        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            print("**First time to train**")

            return False

    def sampling(self):
        dataloader = DataLoader(self.cifar10_train, batch_size=self.batch_size,
                                pin_memory=self.config.pin_memory, sampler=Sampler(self.unlabeled))

        data_dict = {}
        tqdm_batch = tqdm(dataloader, total=len(dataloader))
        total_index = 0
        for curr_it, data in enumerate(tqdm_batch):
            data = data[0].cuda(async=self.config.async_loading)

            quantized, z, distance = self.vae(data, is_train=False)

            for idx in range(quantized.size(0)):
                if quantized[idx] in data_dict:
                    data_dict[quantized[idx]].append([-1*distance[idx], total_index])
                else:
                    data_dict[quantized[idx]] = [[-1 * distance[idx], total_index]]
                total_index += 1

        tqdm_batch.close()

        subset = []
        total_remain = []
        quota = int(self.batch_size / len(data_dict))
        for i in data_dict:
            tmp_list = sorted(data_dict[i], lambda x: x[0], reverse=True)

            if len(tmp_list) > quota:
                subset += list(np.array(tmp_list)[:quota, 2])
                total_remain += tmp_list[quota:]
            else:
                subset += list(np.array(tmp_list)[:, 2])

        if len(subset) < self.budget:
            tmp_list = sorted(total_remain, lambda x: x[0], reverse=True)
            subset += list(np.array(tmp_list)[:(self.budget - len(subset)), 2])

        self.labeled += subset
        self.unlabeled = list(set(self.unlabeled) - set(subset))
