
class Config(object):
    data_name = ''
    initial_size = 100
    budge_size = 10
    budge_max = 1000
    max_cycle = int(budge_max / budge_size)

    root_path = ''
    data_directory = ''
    summary_directory = ''
    checkpoint_directory = ''

    gpu_cnt = 4

    epoch = 500
    batch_size = 16
    learning_rate = 0.001

    pin_memory = True
