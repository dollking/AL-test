import os
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from .graph.resnet import ResNet18 as resnet
from .graph.lossnet import LossNet as lossnet
from .graph.loss import CELoss as loss
from .graph.loss import LossPredLoss as r_loss
from data.sampler import Sampler

from utils.metrics import AverageMeter
from utils.train_utils import set_logger, count_model_prameters

cudnn.benchmark = True


class ClassificationWithLoss(object):
    def __init__(self, config, step_cnt, sample_list):
        self.config = config
        self.step_cnt = step_cnt
        self.best_acc = 0.0

        self.batch_size = self.config.batch_size

        self.logger = set_logger('train_epoch.log')

        # define dataloader
        if 'cifar' in self.config.data_name:
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ])

            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ])

            if self.config.data_name == 'cifar10':
                self.train_dataset = CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                             train=True, download=True, transform=self.train_transform)
                self.test_dataset = CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                            train=False, download=True, transform=self.test_transform)
            elif self.config.data_name == 'cifar100':
                self.train_dataset = CIFAR100(os.path.join(self.config.root_path, self.config.data_directory),
                                             train=True, download=True, transform=self.train_transform)
                self.test_dataset = CIFAR100(os.path.join(self.config.root_path, self.config.data_directory),
                                            train=False, download=True, transform=self.test_transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2,
                                       pin_memory=self.config.pin_memory, sampler=Sampler(sample_list))
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1,
                                      pin_memory=self.config.pin_memory)

        # define models
        self.task = resnet(self.config.num_classes).cuda()
        self.loss_module = lossnet().cuda()

        # define loss
        self.loss = loss().cuda()
        self.r_loss = r_loss().cuda()

        # define optimizer
        self.task_opt = torch.optim.SGD(self.task.parameters(), lr=self.config.learning_rate,
                                        momentum=self.config.momentum, weight_decay=self.config.wdecay)
        self.loss_opt = torch.optim.SGD(self.loss_module.parameters(), lr=self.config.learning_rate,
                                        momentum=self.config.momentum, weight_decay=self.config.wdecay)

        # define optimize scheduler
        self.task_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.task_opt, milestones=self.config.milestones)
        self.loss_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.loss_opt, milestones=self.config.milestones)

        # initialize train counter
        self.epoch = 0
        self.epochl = self.config.epochl

        self.manual_seed = random.randint(10000, 99999)

        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed_all(self.manual_seed)
        random.seed(self.manual_seed)

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.task = nn.DataParallel(self.task, device_ids=gpu_list)
        self.loss_module = nn.DataParallel(self.loss_module, device_ids=gpu_list)

        self.print_train_info()

    def print_train_info(self):
        print("seed: ", self.manual_seed)
        print('Number of generator parameters: {}'.format(count_model_prameters(self.task)))

    def save_checkpoint(self):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_directory, 'task.pth.tar')

        state = {
            'task_state_dict': self.task.state_dict(),
            'loss_state_dict': self.loss_module.state_dict(),
        }

        torch.save(state, tmp_name)

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

    def run(self):
        try:
            if self.step_cnt:
                self.load_checkpoint()
            self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        for _ in range(self.config.epoch):
            self.epoch += 1
            self.train_by_epoch()
            
            self.task_scheduler.step()
            self.loss_scheduler.step()
            
        self.test_by_epoch()

    def train_by_epoch(self):
        tqdm_batch = tqdm(self.train_loader, leave=False, total=len(self.train_loader))

        self.task.train()
        self.loss_module.train()
        avg_loss = AverageMeter()
        for curr_it, data in enumerate(tqdm_batch):
            self.task_opt.zero_grad()
            self.loss_opt.zero_grad()

            inputs = data[0].cuda(async=self.config.async_loading)
            targets = data[1].cuda(async=self.config.async_loading)

            out, features = self.task(inputs)
            target_loss = self.loss(out, targets, 10)

            if self.epoch > self.epochl:
                for idx in range(len(features)):
                    features[idx] = features[idx].detach()

            pred_loss = self.loss_module(features)
            pred_loss = pred_loss.view([-1, ])

            loss = self.r_loss(pred_loss, target_loss) + torch.mean(target_loss)

            loss.backward()
            self.task_opt.step()
            self.loss_opt.step()

            avg_loss.update(loss)

        tqdm_batch.close()

    def test_by_epoch(self):
        with torch.no_grad():
            tqdm_batch = tqdm(self.test_loader, leave=False, total=len(self.test_loader))

            total = 0
            correct = 0
            for curr_it, data in enumerate(tqdm_batch):
                self.task.eval()
                self.loss_module.eval()

                inputs = data[0].cuda(async=self.config.async_loading)
                targets = data[1].cuda(async=self.config.async_loading)
                total += inputs.size(0)

                out, _ = self.task(inputs)
                _, predicted = torch.max(out.data, 1)
                correct += (predicted == targets).sum().item()

            tqdm_batch.close()

            if correct / total > self.best_acc:
                self.best_acc = correct / total
                self.save_checkpoint()
