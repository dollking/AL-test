import torch

from query.query import Query
from query.strategy import Strategy
from task.classification import Classification as Task
from config import Config


if __name__ == '__main__':
    config = Config()
    query = Query(config, 50000)

    for step_cnt in range(config.max_cycle):
        # train a sampling strategy
        torch.cuda.empty_cache()
        strategy = Strategy(config, step_cnt)
        strategy.run()
        
        # take a new sample
        torch.cuda.empty_cache()
        query.sampling()

        # train a task model
        print('step {}: train data count - {}'.format(step_cnt + 1, (step_cnt + 1) * config.budge_size))

        torch.cuda.empty_cache()
        task = Task(config, step_cnt + 1, query.labeled)
        task.run()

        print('step {}: test accuracy - {}'.format(step_cnt + 1, task.best_acc))