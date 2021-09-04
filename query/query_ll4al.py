import os
import random
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10

from task.graph.resnet import ResNet18 as resnet
from task.graph.lossnet import LossNet as lossnet

from data.sampler import Sampler

cudnn.benchmark = True


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

        # define models
        self.task = resnet(self.config.num_classes).cuda()
        self.loss_module = lossnet().cuda()

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.task = nn.DataParallel(self.task, device_ids=gpu_list)
        self.loss_module = nn.DataParallel(self.loss_module, device_ids=gpu_list)

    def load_checkpoint(self):
        try:
            filename = os.path.join(self.config.root_path, self.config.checkpoint_directory, 'task.pth.tar')
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.task.load_state_dict(checkpoint['task_state_dict'])
            self.loss_module.load_state_dict(checkpoint['loss_state_dict'])

        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_directory))
            print("**First time to train**")

    def sampling(self, step_cnt):
        sample_size = self.budget if step_cnt else self.initial_size
        random.shuffle(self.unlabeled)

        if step_cnt:
            self.load_checkpoint()
            subset = self.unlabeled[:sample_size * 10]

            dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                    pin_memory=self.config.pin_memory, sampler=Sampler(subset))
            tqdm_batch = tqdm(dataloader, total=len(dataloader))

            self.task.eval()
            self.loss_module.eval()
            uncertainty = torch.tensor([]).cuda()
            with torch.no_grad():
                for curr_it, data in enumerate(tqdm_batch):
                    data = data[0].cuda(async=self.config.async_loading)

                    _, features = self.task(data)
                    pred_loss = self.loss_module(features)
                    pred_loss = pred_loss.view([-1, ])

                    uncertainty = torch.cat([uncertainty, pred_loss], 0)

                tqdm_batch.close()

            uncertainty = uncertainty.cpu()

            arg = np.argsort(uncertainty)

            sample_set = list(torch.tensor(subset)[arg][-sample_size:].numpy())

        else:
            sample_set = self.unlabeled[:sample_size]

        self.labeled += sample_set
        self.unlabeled = list(set(self.unlabeled) - set(sample_set))
