import os
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from tensorboardX import SummaryWriter

from .graph.resnet import ResNet18 as resnet
from .graph.loss import CELoss as Loss
from data.sampler import Sampler

from utils.metrics import AverageMeter
from utils.train_utils import set_logger, count_model_prameters

cudnn.benchmark = True


class Classification(object):
    def __init__(self, config, step_cnt, sample_list):
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
        self.cifar10_train = CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                     train=True, download=True, transform=self.train_transform)
        self.cifar10_test = CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                    train=False, download=True, transform=self.test_transform)
        
        self.train_loader = DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=2,
                                       pin_memory=self.config.pin_memory, sampler=Sampler(sample_list))
        self.test_loader = DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=1,
                                      pin_memory=self.config.pin_memory)

        # define models
        self.task = resnet().cuda()

        # define loss
        self.loss = Loss().cuda()

        # define optimizer
        self.task_opt = torch.optim.SGD(self.task.parameters(), lr=self.config.learning_rate,
                                        momentum=self.config.momentum, weight_decay=self.config.wdecay)

        # define optimize scheduler
        self.task_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.task_opt, mode='min',
                                                                         factor=0.8, cooldown=8)

        # initialize train counter
        self.epoch = 0

        self.manual_seed = random.randint(10000, 99999)

        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed_all(self.manual_seed)
        random.seed(self.manual_seed)

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.task = nn.DataParallel(self.task, device_ids=gpu_list)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_directory),
                                            comment=f'cifar10_step_{self.step_cnt}')
        self.print_train_info()

    def print_train_info(self):
        print("seed: ", self.manual_seed)
        print('Number of generator parameters: {}'.format(count_model_prameters(self.task)))

    def save_checkpoint(self):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_directory, 'task.pth.tar')

        state = {
            'task_state_dict': self.task.state_dict(),
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
            self.task.train()
            self.task_opt.zero_grad()

            inputs = data[0].cuda(async=self.config.async_loading)
            targets = data[1].cuda(async=self.config.async_loading)

            out = self.task(inputs)

            loss = self.loss(out, targets, 10)

            loss.backward()
            self.task_opt.step()
            avg_loss.update(loss)

        tqdm_batch.close()

        self.task_scheduler.step(avg_loss.val)

        with torch.no_grad():
            tqdm_batch = tqdm(self.test_loader, leave=False, total=len(self.test_loader))

            total = 0
            correct = 0
            avg_loss = AverageMeter()
            for curr_it, data in enumerate(tqdm_batch):
                self.task.eval()

                inputs = data[0].cuda(async=self.config.async_loading)
                targets = data[1].cuda(async=self.config.async_loading)
                total += inputs.size(0)

                out = self.task(inputs)

                loss = self.loss(out, targets, 10)

                _, predicted = torch.max(out.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                avg_loss.update(loss)

            tqdm_batch.close()

            if correct / total > self.best_acc:
                self.best_acc = correct / total
                self.save_checkpoint()
