import random
import torch
from torch.backends import cudnn

from config import Config

cudnn.deterministic = True

random.seed(9410)
torch.manual_seed(9410)
torch.cuda.manual_seed_all(9410)


def main(version):
    config = Config()

    if version == 1:
        from query.strategy.strategy_v1 import Strategy
        strategy = Strategy(config)
        strategy.run()

    elif version == 2:
        from query.strategy.strategy_v2 import Strategy
        from task.classification_loss import ClassificationWithLoss as Task

        strategy = Strategy(config)
        task = Task(config)

        tmp = [i for i in range(50000)]
        random.shuffle(tmp)
        task.run(tmp[:20000])
        strategy.run(task, tmp[:20000])


if __name__ == '__main__':
    main(1)
