import os
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10

from .graph.vae import VAE as vae
from .graph.loss import MSE as Loss
from task.graph.resnet import ResNet18 as resnet

from utils.metrics import AverageMeter, UncertaintyScore
from utils.train_utils import set_logger, count_model_prameters

cudnn.benchmark = True


class Strategy(object):
    def __init__(self, config, step_cnt):
        self.config = config
        self.step_cnt = step_cnt
        self.best = 999999.0

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.batch_size = self.config.batch_size

        self.logger = set_logger('train_epoch.log')

        # define dataloader
        if self.config.data_name == 'cifar10':
            self.train_dataset = CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                         train=True, download=True, transform=self.train_transform)
            self.test_dataset = CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                        train=False, download=True, transform=self.test_transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,
                                       pin_memory=self.config.pin_memory)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1,
                                      pin_memory=self.config.pin_memory)

        self.uncertainty_score = UncertaintyScore()
        # define models
        self.task = resnet().cuda()
        self.vae = vae(self.config.vae_num_hiddens, self.config.vae_num_residual_layers,
                       self.config.vae_num_residual_hiddens, self.config.vae_num_embeddings,
                       self.config.vae_embedding_dim, self.config.vae_commitment_cost, self.config.vae_distance,
                       self.config.vae_decay).cuda()

        # define loss
        self.loss = Loss().cuda()

        # define optimizer
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=self.config.vae_learning_rate, amsgrad=False)

        # define optimize scheduler
        self.vae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.vae_opt, mode='min', factor=0.8, cooldown=8)

        # initialize train counter
        self.epoch = 0

        self.manual_seed = random.randint(10000, 99999)

        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed_all(self.manual_seed)
        random.seed(self.manual_seed)

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.vae = nn.DataParallel(self.vae, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.
        if self.step_cnt:
            self.load_checkpoint()

        self.print_train_info()

    def print_train_info(self):
        print("seed: ", self.manual_seed)
        print('Number of generator parameters: {}'.format(count_model_prameters(self.vae)))

    def load_checkpoint(self):
        filename = os.path.join(self.config.root_path, self.config.checkpoint_directory, 'task.pth.tar')

        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.task.load_state_dict(checkpoint['task_state_dict'])

        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_directory))
            print("**First time to train**")

    def save_checkpoint(self):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_directory, 'vae.pth.tar')

        state = {
            'vae_state_dict': self.vae.state_dict(),
        }

        torch.save(state, tmp_name)

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        for _ in range(self.config.epoch):
            self.epoch += 1
            self.train_by_epoch()

    def train_by_epoch(self):
        tqdm_batch = tqdm(self.train_loader, leave=False, total=len(self.train_loader))

        avg_loss = AverageMeter()
        for curr_it, data in enumerate(tqdm_batch):
            self.task.eval()
            self.vae.train()
            self.vae_opt.zero_grad()

            data = data[0].cuda(async=self.config.async_loading)

            task_output = self.task(data)
            u_score = self.uncertainty_score(task_output)
            vq_loss, data_recon, perplexity, distance = self.vae(data)

            recon_error = self.loss(data_recon, data)
            loss = recon_error + vq_loss
            if self.step_cnt:   # distance loss
                loss += torch.mean(torch.abs(u_score - distance / self.config.vae_distance))

            loss.backward()
            self.vae_opt.step()
            avg_loss.update(loss)

        tqdm_batch.close()

        self.vae_scheduler.step(avg_loss.val)

        with torch.no_grad():
            tqdm_batch = tqdm(self.test_loader, leave=False, total=len(self.test_loader))

            avg_loss = AverageMeter()
            for curr_it, data in enumerate(tqdm_batch):
                self.task.eval()
                self.vae.eval()

                data = data[0].cuda(async=self.config.async_loading)

                task_output = self.task(data)
                u_score = self.uncertainty_score(task_output)
                vq_loss, data_recon, perplexity, distance = self.vae(data)

                recon_error = self.loss(data_recon, data)
                loss = recon_error + vq_loss
                if self.step_cnt:  # distance loss
                    loss += torch.mean(torch.abs(u_score - distance / self.config.vae_distance))

                avg_loss.update(loss)

            tqdm_batch.close()

            if avg_loss.val < self.best:
                self.best = avg_loss.val
                self.save_checkpoint()
            
            if self.epoch % 50 == 1:
                print(f'{self.epoch} - loss: {avg_loss.val} / best: {self.best}')
