import os
import random


class DataManager(object):
    def __init__(self, config):
        self.train_data_path = os.path.join(config.root_path, config.data_directory, config.data_name+'_train_list.txt')
        self.test_data_path = os.path.join(config.root_path, config.data_directory, config.data_name+'_test_list.txt')
        self.initial_size = config.initial_size
        self.budget_size = config.budge_size
        self.budget_max = config.budge_max

        self.data_pool = []     # total data list (train + test)
        self.test_pool = []     # test data list

        self.closed = []  # non-selected train data list
        self.opened = []    # selected train data list

        self.set_data_list()

    def set_data_list(self):
        with open(self.train_data_path) as fp:
            self.data_pool = fp.readlines()

        with open(self.test_data_path) as fp:
            self.test_pool = fp.readlines()

        random.shuffle(self.data_pool)
        self.opened += self.data_pool[:self.initial_size]
        self.closed = self.data_pool[:self.budget_max]
        self.data_pool = self.test_pool



    def open_data(self, lst):
        self.opened += lst

        idx = 0
        while len(self.closed) < idx:
            if self.closed[idx] in lst:
                self.closed.pop(idx)
            else:
                idx += 1
