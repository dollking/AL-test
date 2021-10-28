import os
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10

from query.graph.transformer import Transformer as transformer
from query.graph.loss import MSE as loss
from data.sampler import Sampler

from utils.metrics import AverageMeter
from utils.train_utils import set_logger, count_model_prameters
from tensorboardX import SummaryWriter

cudnn.benchmark = True


class Strategy(object):
    def __init__(self, config):
        self.config = config
        self.best = 999999.0

        self.batch_size = self.config.batch_size * 8

        self.logger = set_logger('train_epoch.log')

        # define dataloader
        if 'cifar' in self.config.data_name:
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                transforms.RandomErasing(p=0.6, scale=(0.03, 0.08), ratio=(0.3, 3.3)),
            ])
            if self.config.data_name == 'cifar10':
                self.train_dataset = CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                             train=True, download=True, transform=self.train_transform)
            elif self.config.data_name == 'cifar100':
                self.train_dataset = CIFAR100(os.path.join(self.config.root_path, self.config.data_directory),
                                              train=True, download=True, transform=self.train_transform)

        # define models
        self.transformer = transformer(self.config.vae_embedding_dim).cuda()

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.transformer = nn.DataParallel(self.transformer, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.

        self.print_train_info()
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_directory),
                                            comment='Transformer')

    def print_train_info(self):
        print('Number of generator parameters: {}'.format(count_model_prameters(self.transformer)))

    def save_checkpoint(self):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_directory, 'transformer.pth.tar')

        state = {
            'transformer_state_dict': self.transformer.state_dict(),
        }

        torch.save(state, tmp_name)

    def set_train(self):
        # define loss
        self.loss = loss().cuda()

        # define optimizer
        self.transformer_opt = torch.optim.Adam(self.transformer.parameters(), lr=self.config.vae_learning_rate)

        # define optimize scheduler
        self.transformer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.transformer_opt, mode='min',
                                                                                factor=0.8, cooldown=8)

        # initialize train counter
        self.epoch = 0

    def run(self, task, ae, sample_list):
        data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2,
                                 pin_memory=self.config.pin_memory, sampler=Sampler(sample_list))
        try:
            self.set_train()
            self.train(task, ae, data_loader)

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self, task, ae, data_loader):
        for _ in range(self.config.vae_epoch):
            self.epoch += 1
            self.train_by_epoch(task, ae, data_loader)
        self.save_checkpoint()

    def train_by_epoch(self, task, ae, data_loader):
        tqdm_batch = tqdm(data_loader, leave=False, total=len(data_loader))

        avg_loss = AverageMeter()

        self.transformer.train()
        for curr_it, data in enumerate(tqdm_batch):
            self.transformer_opt.zero_grad()

            data = data[0].cuda(async=self.config.async_loading)

            task_features = task.get_feature(data)
            for idx in range(len(task_features)):
                task_features[idx] = task_features[idx].detach()
            pre_features = self.transformer(task_features)

            ae_features = ae.get_feature(data)
            ae_features = ae_features.detach()

            # reconstruction loss
            loss = self.loss(pre_features, ae_features.view([-1, self.config.vae_embedding_dim]))

            loss.backward()
            self.transformer_opt.step()

            avg_loss.update(loss)

        tqdm_batch.close()
        self.transformer_scheduler.step(avg_loss.val)

        self.summary_writer.add_scalar('loss', avg_loss.val, self.epoch)

        if self.epoch % 50 == 0:
            print(f'{self.epoch} - loss: {avg_loss.val}')

    def get_feature(self, task_features):
        self.transformer.eval()
        with torch.no_grad():
            features = self.transformer(task_features)

        return features
