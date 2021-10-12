import os
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from query.graph.ae_sign import AE as ae
from query.graph.loss import MSE as loss

from torchvision.datasets import CIFAR10, CIFAR100

from utils.metrics import AverageMeter, mAP
from utils.train_utils import set_logger, count_model_prameters

from tensorboardX import SummaryWriter

cudnn.benchmark = False


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
                self.test_dataset = CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                            train=False, download=True, transform=self.train_transform)
            elif self.config.data_name == 'cifar100':
                self.train_dataset = CIFAR100(os.path.join(self.config.root_path, self.config.data_directory),
                                              train=True, download=True, transform=self.train_transform)
                self.test_dataset = CIFAR100(os.path.join(self.config.root_path, self.config.data_directory),
                                             train=False, download=True, transform=self.train_transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,
                                       pin_memory=self.config.pin_memory)

        # define models
        self.ae = ae(self.config.vae_num_hiddens, self.config.vae_num_residual_layers,
                       self.config.vae_num_residual_hiddens, self.config.vae_embedding_dim).cuda()

        # define loss
        self.loss = loss().cuda()

        # define optimizer
        self.ae_opt = torch.optim.Adam(self.ae.parameters(), lr=self.config.vae_learning_rate)

        # define optimize scheduler
        self.ae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.ae_opt, mode='min', factor=0.8,
                                                                       cooldown=8)

        # initialize train counter
        self.epoch = 0

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.ae = nn.DataParallel(self.ae, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.

        self.print_train_info()
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_directory),
                                            comment='AE-CODE')

    def print_train_info(self):
        print('Number of generator parameters: {}'.format(count_model_prameters(self.ae)))

    def save_checkpoint(self):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_directory, 'ae.pth.tar')

        state = {
            'ae_state_dict': self.ae.state_dict(),
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

        self.ae.train()
        for curr_it, data in enumerate(tqdm_batch):
            self.ae_opt.zero_grad()

            inputs = data[0].cuda(async=self.config.async_loading)

            recon, code = self.ae(inputs)

            loss = self.loss(recon, inputs)

            loss.backward()
            self.ae_opt.step()

            avg_loss.update(loss)

            if self.epoch % 50 == 0:
                centroid_set |= set(tuple(map(tuple, code.view([-1, self.config.vae_embedding_dim]).cpu().tolist())))

        tqdm_batch.close()
        self.ae_scheduler.step(avg_loss.val)

        self.summary_writer.add_image('image/origin', inputs[0], self.epoch)
        self.summary_writer.add_image('image/recon_origin', recon[0], self.epoch)

        self.summary_writer.add_scalar("loss", avg_loss.val, self.epoch)

        if self.epoch % 50 == 0:
            print(f'{self.epoch} - loss: {avg_loss.val} / best: {self.best} / centroid cnt: {len(centroid_set)}')

    def get_code(self, inputs):
        self.ae.eval()
        with torch.no_grad():
            _, code = self.ae(inputs)

        return code
