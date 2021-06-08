
class Config(object):
    data_name = 'cifar10'
    initial_size = 100
    budge_size = 10
    budge_max = 1000
    max_cycle = int(budge_max / budge_size)

    root_path = '/home/D2019063/Active_Learning/'
    data_directory = 'data'
    summary_directory = 'board'
    checkpoint_directory = 'model'

    gpu_cnt = 4

    epoch = 500
    batch_size = 16
    learning_rate = 0.001

    pin_memory = True
    async_loading = True
