import os
import shutil
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from tensorboardX import SummaryWriter

from .graph.resnet import ResNet18 as Model
from .graph.resnet import Loss
from .dataset.classification import ClassificationDataset

from utils.metrics import AverageMeter
from utils.train_utils import set_logger, count_model_prameters

cudnn.benchmark = True


class Cifar10(object):
    def __init__(self, config, data_manager, step_cnt, is_continue=False):
        self.config = config
        self.data_manager = data_manager
        self.step_cnt = step_cnt

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.batch_size = self.config.batch_size

        self.logger = set_logger('train_epoch.log')

        # define dataloader
        self.dataset = ClassificationDataset(self.config, self.transform, self.data_manager.opened)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                                     pin_memory=self.config.pin_memory, collate_fn=self.collate_function)

        self.dataset_test = ClassificationDataset(self.config, self.transform, self.data_manager.test_pool)
        self.testloader = DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                     pin_memory=self.config.pin_memory, collate_fn=self.collate_function)

        # define models
        self.model = Model().cuda()

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
        self.accumulate_iter = 0
        self.total_iter = (len(self.dataset) + self.config.batch_size - 1) // self.config.batch_size

        self.manual_seed = random.randint(10000, 99999)

        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed_all(self.manual_seed)
        random.seed(self.manual_seed)

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.model = nn.DataParallel(self.model, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.
        if is_continue:
            self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_directory),
                                            comment=f'cifar10_step_{self.step_cnt}')
        self.print_train_info()

    def print_train_info(self):
        print("seed: ", self.manual_seed)
        print('Number of generator parameters: {}'.format(count_model_prameters(self.model)))

    def collate_function(self, samples):
        X = torch.cat([sample['X'] for sample in samples], axis=0)
        target = torch.cat([sample['target'] for sample in samples], axis=0)

        return tuple([X, target])

    def load_checkpoint(self, file_name):
        filename = os.path.join(self.config.root_path, self.config.checkpoint_dir, file_name)
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.model.load_state_dict(checkpoint['model_state_dict'])

        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            print("**First time to train**")

    def save_checkpoint(self, epoch):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                'checkpoint_{}.pth.tar'.format(epoch))

        state = {
            'model_state_dict': self.model.state_dict(),
        }

        torch.save(state, tmp_name)
        shutil.copyfile(tmp_name, os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                               self.config.checkpoint_file))

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def record_image(self, X, out, target, step='train'):
        self.summary_writer.add_image(step + '/img 1', X[0], self.epoch)
        self.summary_writer.add_image(step + '/img 2', X[1], self.epoch)
        self.summary_writer.add_image(step + '/img 3', X[2], self.epoch)

        self.summary_writer.add_image(step + '/result 1', out[0], self.epoch)
        self.summary_writer.add_image(step + '/result 2', out[1], self.epoch)
        self.summary_writer.add_image(step + '/result 3', out[2], self.epoch)

        self.summary_writer.add_image(step + '/target 1', target[0], self.epoch)
        self.summary_writer.add_image(step + '/target 2', target[1], self.epoch)
        self.summary_writer.add_image(step + '/target 3', target[2], self.epoch)

    def train(self):
        for _ in range(self.config.epoch):
            self.epoch += 1
            self.train_by_epoch()

            self.save_checkpoint(self.config.checkpoint_file)

    def train_by_epoch(self):
        tqdm_batch = tqdm(self.dataloader, total=self.total_iter, desc="epoch-{}".format(self.epoch))

        avg_loss = AverageMeter()
        for curr_it, (X, target) in enumerate(tqdm_batch):
            self.model.train()
            self.opt.zero_grad()

            X = X.cuda(async=self.config.async_loading)
            target = target.cuda(async=self.config.async_loading)

            logit = self.model(X)

            loss = self.loss(logit, target)

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

            avg_loss = AverageMeter()
            for curr_it, (X, target) in enumerate(tqdm_batch):
                self.model.eval()

                X = X.cuda(async=self.config.async_loading)
                target = target.cuda(async=self.config.async_loading)

                logit = self.model(X)

                loss = self.loss(logit, target)

                avg_loss.update(loss)

            tqdm_batch.close()

            self.summary_writer.add_scalar('eval/loss', avg_loss.val, self.epoch)
