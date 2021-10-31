import os
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from .graph.resnet import ResNet18 as resnet
from .graph.transformer import Transformer as transformer
from .graph.loss import CELoss as loss
from .graph.loss import MSE as mse_loss
from data.sampler import Sampler

from utils.metrics import AverageMeter
from utils.train_utils import count_model_prameters

cudnn.benchmark = False


class ClassificationWithTrans(object):
    def __init__(self, config):
        self.config = config
        self.best_acc = 0.0

        self.batch_size = self.config.batch_size

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

        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1,
                                      pin_memory=self.config.pin_memory)

        # define models
        self.task = resnet(self.config.num_classes).cuda()
        self.transformer = transformer(config.vae_embedding_dim).cuda()

        self.epochl = self.config.epochl

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.task = nn.DataParallel(self.task, device_ids=gpu_list)
        self.transformer = nn.DataParallel(self.transformer, device_ids=gpu_list)

        self.print_train_info()

    def print_train_info(self):
        print('Number of generator parameters: {}'.format(count_model_prameters(self.task)))

    def save_checkpoint(self):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_directory, 'task.pth.tar')

        state = {
            'task_state_dict': self.task.state_dict(),
            'transformer_state_dict': self.transformer.state_dict(),
        }

        torch.save(state, tmp_name)

    def set_train(self):
        # define loss
        self.loss = loss().cuda()
        self.mse = mse_loss().cuda()

        # define optimizer
        self.task_opt = torch.optim.SGD(self.task.parameters(), lr=self.config.learning_rate,
                                        momentum=self.config.momentum, weight_decay=self.config.wdecay)
        self.transformer_opt = torch.optim.SGD(self.transformer.parameters(), lr=self.config.learning_rate,
                                        momentum=self.config.momentum, weight_decay=self.config.wdecay)

        # define optimize scheduler
        self.task_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.task_opt, milestones=self.config.milestones)
        self.transformer_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.transformer_opt, milestones=self.config.milestones)

        # initialize train counter
        self.epoch = 0

    def run(self, sample_list, ae):
        data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2,
                                 pin_memory=self.config.pin_memory, sampler=Sampler(sample_list))
        try:
            self.set_train()
            self.train(data_loader, ae)

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self, data_loader, ae):
        for _ in range(self.config.epoch):
            self.epoch += 1
            self.train_by_epoch(data_loader, ae)

            self.task_scheduler.step()
            self.transformer_scheduler.step()

        self.test()

    def train_by_epoch(self, data_loader, ae):
        tqdm_batch = tqdm(data_loader, leave=False, total=len(data_loader))

        self.task.train()
        self.transformer.train()
        avg_loss = AverageMeter()
        for curr_it, data in enumerate(tqdm_batch):
            self.task_opt.zero_grad()
            self.transformer_opt.zero_grad()

            inputs = data[0].cuda(async=self.config.async_loading)
            targets = data[1].cuda(async=self.config.async_loading)

            out, features = self.task(inputs)
            target_loss = self.loss(out, targets, 10)

            if self.epoch > self.epochl:
                for idx in range(len(features)):
                    features[idx] = features[idx].detach()

            pre_features = self.transformer(features)

            ae_features = ae.get_feature(inputs)
            ae_features = ae_features.view([-1, self.config.vae_embedding_dim])
            ae_features = ae_features.detach()

            loss = self.mse(pre_features, ae_features) + torch.mean(target_loss)

            loss.backward()
            self.task_opt.step()
            self.transformer_opt.step()

            avg_loss.update(loss)

        tqdm_batch.close()

    def test(self):
        with torch.no_grad():
            tqdm_batch = tqdm(self.test_loader, leave=False, total=len(self.test_loader))

            total = 0
            correct = 0
            for curr_it, data in enumerate(tqdm_batch):
                self.task.eval()
                self.transformer.eval()

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

    def get_feature(self, inputs):
        self.task.eval()
        self.transformer.eval()
        with torch.no_grad():
            inputs = inputs.cuda(async=self.config.async_loading)

            out, features = self.task(inputs)
            pred_features = self.transformer(features)

        return pred_features

    def get_result2(self, inputs, targets):
        self.task.eval()
        self.transformer.eval()
        with torch.no_grad():
            inputs = inputs.cuda(async=self.config.async_loading)

            out, features = self.task(inputs)

            loss = self.loss(out, targets, 10)
            loss = loss.view([-1, ])

        return out, features, loss
