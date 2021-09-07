from config import Config
from query.query_ll4al import Query
from task.classification_loss import ClassificationWithLoss as Task


def main(cycle_cnt):
    config = Config()
    query = Query(config)
    task = Task(config)

    fp = open(f'record_{cycle_cnt}.txt', 'w')
    for step_cnt in range(config.max_cycle):
        # take a new sample
        query.sampling(step_cnt, task)

        # train a task model
        print('step {}: train data count - {}'.format(step_cnt + 1, len(set(query.labeled))))

        task.run(query.labeled)

        print('step {}: test accuracy - {}'.format(step_cnt + 1, task.best_acc))

        fp.write(f'{task.best_acc}\n')

    fp.close()


if __name__ == '__main__':
    for i in range(10):
        main(i)
