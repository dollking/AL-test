import torch

from query.query_v3 import Query
from query.strategy.strategy_v3 import Strategy
from task.classification_loss import ClassificationWithLoss as Task
from config import Config


if __name__ == '__main__':
    config = Config()
    query = Query(config)

    fp = open('record.txt', 'w')
    for step_cnt in range(config.max_cycle):
        # take a new sample
        query.sampling(step_cnt)

        # train a task model
        print('step {}: train data count - {}'.format(step_cnt + 1, len(query.labeled)))

        task = Task(config, step_cnt + 1, query.labeled)
        task.run()

        print('step {}: test accuracy - {}'.format(step_cnt + 1, task.best_acc))

        fp.write(f'{task.best_acc}\n')

    fp.close()
