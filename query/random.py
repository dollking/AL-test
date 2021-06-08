import os
import random


class Random(object):
    def __init__(self, closed_list, config):
        self.close_list = closed_list
        self.budget_size = config.budge_size

    def select_data(self):
        return random.sample(self.close_list, self.budget_size)
