from data.manager import DataManager
from query.random import Random as Query
from task.cifar10 import Resnet as Task
from config import Config


if __name__ == '__main__':
    config = Config()

    manager = DataManager(config)
    task = Task()
    query = Query(manager.closed, config)


