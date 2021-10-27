import random
import numpy as np

import torch
from torch.backends import cudnn

from config import Config
from query.strategy.strategy_ae import Strategy as autoencoder
from query.strategy.strategy_transformer import Strategy as transformer
from task.classification_loss import ClassificationWithLoss as Task

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_strategy(cycle_cnt):
    torch.manual_seed(cycle_cnt * 1000)
    torch.cuda.manual_seed_all(cycle_cnt * 1000)

    config = Config()
    ae = autoencoder(config)

    ae.run()

    return ae


def main(cycle_cnt, config, query, ae):
    random.seed(cycle_cnt * 1000)
    np.random.seed(cycle_cnt * 1000)
    torch.manual_seed(cycle_cnt * 1000)
    torch.cuda.manual_seed_all(cycle_cnt * 1000)

    task = Task(config)
    trans = transformer(config)

    # train a task model
    task.run([i for i in range(50000)])



    print(f'trial-{cycle_cnt} / step {step_cnt + 1}: train data count - {len(set(query.labeled))}')
    print(f'test accuracy - {task.best_acc}')


if __name__ == '__main__':
    for i in range(1):
        ae = train_strategy(i + 1)
        config = Config()
        main(i + 1, config, ae)