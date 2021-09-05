import os
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10

from query.graph.vae_v5 import VAE as vae
from query.graph.loss import MSE as loss
from query.graph.loss import SelfClusteringLoss as scloss
from task.graph.resnet import ResNet18 as resnet
from data.dataset import Dataset_CIFAR10, Dataset_CIFAR100

from utils.metrics import AverageMeter, UncertaintyScore
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
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ])
            if self.config.data_name == 'cifar10':
                self.train_dataset = CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                             train=True, download=True, transform=self.train_transform)
            elif self.config.data_name == 'cifar100':
                self.train_dataset = CIFAR100(os.path.join(self.config.root_path, self.config.data_directory),
                                              train=True, download=True, transform=self.train_transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,
                                       pin_memory=self.config.pin_memory)

        # define models
        self.task = resnet(self.config.num_classes).cuda()
        self.vae = vae(self.config.vae_num_hiddens, self.config.vae_num_residual_layers,
                       self.config.vae_num_residual_hiddens, self.config.vae_num_embeddings,
                       self.config.vae_embedding_dim, self.config.vae_commitment_cost, self.config.vae_distance,
                       self.config.vae_decay).cuda()

        # define loss
        self.loss = loss().cuda()
        self.scloss = scloss().cuda()

        # define optimizer
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=self.config.vae_learning_rate)

        # define optimize scheduler
        self.vae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.vae_opt, mode='min', factor=0.8,
                                                                        cooldown=8)

        # initialize train counter
        self.epoch = 0

        self.manual_seed = random.randint(10000, 99999)

        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed_all(self.manual_seed)
        random.seed(self.manual_seed)

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.vae = nn.DataParallel(self.vae, device_ids=gpu_list)
        self.task = nn.DataParallel(self.task, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.

        self.print_train_info()
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_directory),
                                            comment='VQ-VAE')

    def print_train_info(self):
        print("seed: ", self.manual_seed)
        print('Number of generator parameters: {}'.format(count_model_prameters(self.vae)))

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
        for _ in range(self.config.vae_epoch):
            self.epoch += 1
            self.train_by_epoch()
        self.save_checkpoint()

    def train_by_epoch(self):
        tqdm_batch = tqdm(self.train_loader, leave=False, total=len(self.train_loader))

        centroid_set = set()
        avg_loss = AverageMeter()
        for curr_it, data in enumerate(tqdm_batch):
            self.vae.train()
            self.vae_opt.zero_grad()

            data = data[0].cuda(async=self.config.async_loading)
            if self.epoch % 3:
                _, data_recon, _ = self.vae(data, False)

                recon_loss = self.loss(data_recon, data)
                loss = recon_loss
            else:
                vq_loss, data_recon, encoding_indices = self.vae(data)

                recon_loss = self.loss(data_recon, data)
                loss = recon_loss + vq_loss

                centroid_set |= set(encoding_indices.view([-1, ]).cpu().tolist())

            loss.backward()
            self.vae_opt.step()

            avg_loss.update(loss)

        tqdm_batch.close()
        self.vae_scheduler.step(avg_loss.val)

        self.summary_writer.add_image('image/origin', data[0], self.epoch)
        self.summary_writer.add_image('image/recon_origin', data[0], self.epoch)
        self.summary_writer.add_scalar("loss", avg_loss.val, self.epoch)

        if self.epoch % 30 == 0:
            print(f'{self.epoch} - loss: {avg_loss.val} / best: {self.best} / centroid cnt: {len(centroid_set)}')
