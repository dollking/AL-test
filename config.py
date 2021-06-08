
class Config(object):
    data_name = ''

    root_path = ''
    data_directory = ''
    summary_directory = ''
    checkpoint_directory = ''

    gpu_cnt = 4

    epoch = 500
    batch_size = 16
    learning_rate = 0.001

    pin_memory = True

    budge_size = 10
    budge_max = 10000
    max_cycle = budge_max / budge_size
