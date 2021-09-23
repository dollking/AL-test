import os
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from query.graph.hash import Hash as hashnet
from query.graph.loss import HashLoss as hloss
from query.graph.loss import CodeLoss as closs
from data.dataset import Dataset_CIFAR10, Dataset_CIFAR100
from data.sampler import Sampler

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
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                transforms.RandomErasing(p=0.6, scale=(0.03, 0.08), ratio=(0.3, 3.3)),
            ])

            if self.config.data_name == 'cifar10':
                self.train_dataset = Dataset_CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                                     train=True, download=True, transform=self.train_transform)
                self.test_dataset = Dataset_CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                                    train=False, download=True, transform=self.train_transform)
            elif self.config.data_name == 'cifar100':
                self.train_dataset = Dataset_CIFAR100(os.path.join(self.config.root_path, self.config.data_directory),
                                                      train=True, download=True, transform=self.train_transform)
                self.test_dataset = Dataset_CIFAR100(os.path.join(self.config.root_path, self.config.data_directory),
                                                     train=False, download=True, transform=self.train_transform)

        # define models
        self.hashnet = hashnet(self.config.vae_embedding_dim).cuda()

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.hashnet = nn.DataParallel(self.hashnet, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.

        self.print_train_info()
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_directory),
                                            comment='AE-HASH')

    def print_train_info(self):
        print('Number of generator parameters: {}'.format(count_model_prameters(self.hashnet)))

    def save_checkpoint(self):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_directory, 'hashnet.pth.tar')

        state = {
            'hashnet_state_dict': self.hashnet.state_dict(),
        }

        torch.save(state, tmp_name)

    def set_train(self):
        # define loss
        self.hloss = hloss().cuda()
        self.closs = closs().cuda()

        # define optimizer
        self.hashnet_opt = torch.optim.Adam(self.hashnet.parameters(), lr=self.config.vae_learning_rate)

        # define optimize scheduler
        self.hashnet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.hashnet_opt, mode='min', factor=0.8,
                                                                            cooldown=8)

        # initialize train counter
        self.epoch = 0

    def run(self, task, sample_list):
        try:
            self.set_train()
            self.train(task, sample_list)

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self, task, sample_list):
        for _ in range(self.config.vae_epoch):
            self.epoch += 1
            self.train_by_epoch(task, sample_list[:])

        self.test(task)
        self.save_checkpoint()

    def train_by_epoch(self, task, sample_list):
        if self.epoch % 2:
            random.shuffle(sample_list)
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2,
                                      pin_memory=self.config.pin_memory, sampler=Sampler(sample_list))
        else:
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2, shuffle=True,
                                      pin_memory=self.config.pin_memory)

        tqdm_batch = tqdm(train_loader, leave=False, total=len(train_loader))

        centroid_set = set()
        avg_loss = AverageMeter()
        avg_code_loss = AverageMeter()
        avg_balance_loss = AverageMeter()

        self.hashnet.train()
        for curr_it, data in enumerate(tqdm_batch):
            self.hashnet_opt.zero_grad()

            origin_data = data['origin'].cuda(async=self.config.async_loading)
            trans_data = data['trans'].cuda(async=self.config.async_loading)
            target = data['target'].cuda(async=self.config.async_loading)

            _, origin_features, _ = task.get_result(origin_data)
            origin_logit = self.hashnet(origin_features)

            _, trans_features, _ = task.get_result(trans_data)
            trans_logit = self.hashnet(trans_features)

            code_balance_loss, code_loss = self.closs(origin_logit, trans_logit)
            loss = code_balance_loss + code_loss * 0.1

            if self.epoch % 2:
                hash_loss = self.hloss(trans_logit, target, self.config.vae_embedding_dim * 2)
                loss = hash_loss + loss * 0.1

            loss.backward()
            self.hashnet_opt.step()

            if self.epoch % 2:
                avg_loss.update(loss)
                avg_code_loss.update(code_loss)
                avg_balance_loss.update(code_balance_loss)

                origin_code = torch.sign(origin_logit)
                centroid_set |= set(tuple(map(tuple, origin_code.view([-1, self.config.vae_embedding_dim]).cpu().tolist())))

        tqdm_batch.close()
        self.hashnet_scheduler.step(avg_loss.val)

        if self.epoch % 2:
            self.summary_writer.add_scalar("loss", avg_loss.val, self.epoch)
            self.summary_writer.add_scalar("balance_loss", avg_balance_loss.val, self.epoch)
            self.summary_writer.add_scalar("code_loss", avg_code_loss.val, self.epoch)

        if self.epoch % 50 == 49:
            print(f'{self.epoch} - loss: {avg_loss.val} / best: {self.best} / centroid cnt: {len(centroid_set)}')

    def test(self, task):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2,
                                  pin_memory=self.config.pin_memory)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2,
                                 pin_memory=self.config.pin_memory)

        train_code, test_code, train_label, test_label = [], [], [], []
        self.hashnet.eval()
        with torch.no_grad():
            tqdm_train = tqdm(train_loader, leave=False, total=len(train_loader))
            for curr_it, data in enumerate(tqdm_train):
                origin_data = data['origin'].cuda(async=self.config.async_loading)
                target = data['target'].cuda(async=self.config.async_loading)

                _, origin_features, _ = task.get_result(origin_data)
                origin_logit = self.hashnet(origin_features)

                train_code.append(torch.sign(origin_logit))
                train_label.append(target)

            tqdm_train.close()
            train_code, train_label = torch.cat(train_code), torch.cat(train_label)

            tqdm_test = tqdm(test_loader, leave=False, total=len(test_loader))
            for curr_it, data in enumerate(tqdm_test):
                origin_data = data['origin'].cuda(async=self.config.async_loading)
                target = data['target'].cuda(async=self.config.async_loading)

                _, origin_features, _ = task.get_result(origin_data)
                origin_logit = self.hashnet(origin_features)

                test_code.append(torch.sign(origin_logit))
                test_label.append(target)

            tqdm_test.close()
            test_code, test_label = torch.cat(test_code), torch.cat(test_label)

        map = mAP(train_code, test_code, train_label, test_label)
        print(f'--- retrieval mAP: {map} ---')

    def get_code(self, inputs):
        self.hashnet.eval()
        with torch.no_grad():
            logit = self.hashnet(inputs)

        return torch.sign(logit)
