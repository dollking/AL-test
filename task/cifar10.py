import os
import shutil
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10

from tensorboardX import SummaryWriter

from .graph.resnet import ResNet18 as resnet
from .graph.resnet import Loss
from .sampler import Sampler

from utils.metrics import AverageMeter
from utils.train_utils import set_logger, count_model_prameters

cudnn.benchmark = True


class Cifar10(object):
    def __init__(self, config, step_cnt, is_continue=False):
        self.config = config
        self.step_cnt = step_cnt
        self.best_acc = 0.0

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
        cifar10_train = CIFAR10('data/cifar10', train=True, download=True, transform=self.train_transform)
        cifar10_unlabeled = CIFAR10('data/cifar10', train=True, download=True, transform=self.test_transform)
        cifar10_test = CIFAR10('data/cifar10', train=False, download=True, transform=self.test_transform)

        self.train_loader = DataLoader(cifar10_train, batch_size=self.batch_size, shuffle=False, num_workers=2,
                                       pin_memory=self.config.pin_memory, sampler=Sampler())
        self.test_loader = DataLoader(cifar10_test, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                      pin_memory=self.config.pin_memory, sampler=Sampler())
        self.unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                           pin_memory=self.config.pin_memory, sampler=Sampler())


        # define models
        self.model = resnet().cuda()

        # define loss
        self.loss = Loss().cuda()

        # define lr
        self.lr = self.config.learning_rate

        # define optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # define optimize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.8, cooldown=20)

        # initialize train counter
        self.epoch = 0

        self.manual_seed = random.randint(10000, 99999)

        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed_all(self.manual_seed)
        random.seed(self.manual_seed)

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.model = nn.DataParallel(self.model, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.
        if is_continue:
            self.load_checkpoint()

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_directory),
                                            comment=f'cifar10_step_{self.step_cnt}')
        self.print_train_info()

    def print_train_info(self):
        print("seed: ", self.manual_seed)
        print('Number of generator parameters: {}'.format(count_model_prameters(self.model)))

    def load_checkpoint(self):
        filename = os.path.join(self.config.root_path, self.config.checkpoint_directory,
                                '{}_step_{}.pth.tar'.format(self.config.data_name, self.step_cnt-1))
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.model.load_state_dict(checkpoint['model_state_dict'])

        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            print("**First time to train**")

    def save_checkpoint(self):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_directory,
                                '{}_step_{}.pth.tar'.format(self.config.data_name, self.step_cnt))

        state = {
            'model_state_dict': self.model.state_dict(),
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
        tqdm_batch = tqdm(self.dataloader,
                          total=(len(self.dataset) + self.config.batch_size - 1) // self.config.batch_size,
                          desc="epoch-{}".format(self.epoch))

        avg_loss = AverageMeter()
        for curr_it, (X, target) in enumerate(tqdm_batch):
            self.model.train()
            self.opt.zero_grad()

            X = X.cuda(async=self.config.async_loading)
            target = target.cuda(async=self.config.async_loading)

            logit = self.model(X)

            loss = self.loss(logit, target, 10)

            loss.backward()
            self.opt.step()
            avg_loss.update(loss)

        tqdm_batch.close()

        self.summary_writer.add_scalar('train/loss', avg_loss.val, self.epoch)
        self.scheduler.step(avg_loss.val)

        with torch.no_grad():
            tqdm_batch = tqdm(self.testloader,
                              total=(len(self.dataset_test) + self.config.batch_size - 1) // self.config.batch_size,
                              desc="epoch-{}".format(self.epoch))

            total = 0
            correct = 0
            avg_loss = AverageMeter()
            for curr_it, (X, target) in enumerate(tqdm_batch):
                self.model.eval()

                X = X.cuda(async=self.config.async_loading)
                target = target.cuda(async=self.config.async_loading)
                logit = self.model(X)
                loss = self.loss(logit, target, 10)

                _, predicted = torch.max(logit.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                avg_loss.update(loss)

            tqdm_batch.close()

            self.summary_writer.add_scalar('eval/loss', avg_loss.val, self.epoch)

            if correct / total > self.best_acc:
                self.best_acc = correct / total
                self.save_checkpoint()
