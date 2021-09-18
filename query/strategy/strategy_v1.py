import os
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from query.graph.vae import VAE as vae
from query.graph.loss import MSE as loss
from query.graph.loss import CodeLoss as closs
from query.graph.loss import KldLoss as kloss

from data.dataset import Dataset_CIFAR10, Dataset_CIFAR100

from utils.metrics import AverageMeter
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
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                transforms.RandomErasing(p=0.6, scale=(0.03, 0.08), ratio=(0.3, 3.3)),
            ])

            if self.config.data_name == 'cifar10':
                self.train_dataset = Dataset_CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                                     train=True, download=True, transform=self.train_transform)
            elif self.config.data_name == 'cifar100':
                self.train_dataset = Dataset_CIFAR100(os.path.join(self.config.root_path, self.config.data_directory),
                                                      train=True, download=True, transform=self.train_transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,
                                       pin_memory=self.config.pin_memory)

        # define models
        self.vae = vae(self.config.vae_num_hiddens, self.config.vae_num_residual_layers,
                       self.config.vae_num_residual_hiddens, self.config.vae_num_embeddings,
                       self.config.vae_embedding_dim, self.config.vae_distance).cuda()

        # define loss
        self.loss = loss().cuda()
        self.closs = closs().cuda()
        self.kloss = kloss().cuda()

        # define optimizer
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=self.config.vae_learning_rate)

        # define optimize scheduler
        self.vae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.vae_opt, mode='min', factor=0.8,
                                                                        cooldown=8)

        # initialize train counter
        self.epoch = 0

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.vae = nn.DataParallel(self.vae, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.

        self.print_train_info()
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_directory),
                                            comment='AE-HASH')

    def print_train_info(self):
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
        avg_code_loss = AverageMeter()
        avg_balance_loss = AverageMeter()

        self.vae.train()
        for curr_it, data in enumerate(tqdm_batch):
            self.vae_opt.zero_grad()

            origin_data = data['origin'].cuda(async=self.config.async_loading)
            trans_data = data['trans'].cuda(async=self.config.async_loading)

            origin_recon, origin_logit, origin_mu, origin_logvar = self.vae(origin_data)
            trans_recon, trans_logit, trans_mu, trans_logvar = self.vae(trans_data)

            # reconstruction loss
            recon_loss = (self.loss(origin_recon, origin_data) + self.loss(trans_recon, trans_data)) / 2
            kld_loss = (self.kloss(origin_mu, origin_logvar) + self.kloss(trans_mu, trans_logvar)) / 2
            _, code_loss = self.closs(origin_logit, trans_logit)

            loss = (recon_loss + 0.1 * kld_loss) + code_loss

            loss.backward()
            self.vae_opt.step()

            avg_loss.update(loss)
            avg_code_loss.update(code_loss)
            # avg_balance_loss.update(code_balance_loss)

            if self.epoch % 50 == 0:
                origin_code = torch.sign(origin_logit)
                centroid_set |= set(tuple(map(tuple, origin_code.view([-1, self.config.vae_embedding_dim]).cpu().tolist())))

        tqdm_batch.close()
        self.vae_scheduler.step(avg_loss.val)

        self.summary_writer.add_image('image/origin', origin_data[0], self.epoch)
        self.summary_writer.add_image('image/trans', trans_data[0], self.epoch)
        self.summary_writer.add_image('image/recon_origin', origin_recon[0], self.epoch)

        self.summary_writer.add_scalar("loss", avg_loss.val, self.epoch)
        self.summary_writer.add_scalar("balance_loss", avg_balance_loss.val, self.epoch)
        self.summary_writer.add_scalar("code_loss", avg_code_loss.val, self.epoch)

        if self.epoch % 50 == 0:
            print(f'{self.epoch} - loss: {avg_loss.val} / best: {self.best} / centroid cnt: {len(centroid_set)}')

    def get_code(self, inputs):
        self.vae.eval()
        with torch.no_grad():
            _, code, _, _ = self.vae(inputs)

        return torch.sign(code)
