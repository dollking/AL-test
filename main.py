from data.manager import DataManager
from query.random import Random as Query
from task.cifar10 import Cifar10 as Task
from config import Config


if __name__ == '__main__':
    config = Config()

    manager = DataManager(config)
    query = Query(manager.closed, config)

    for step_cnt in range(config.max_cycle):
        task = Task(config, manager, step_cnt + 1)
        task.run()
