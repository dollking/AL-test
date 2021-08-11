
class Config(object):
    data_name = 'cifar10'
    data_size = 50000

    budge_size = 1000
    budge_max = 10000

    max_cycle = int(budge_max / budge_size)

    root_path = '/home/D2019063/Active_Learning/'
    data_directory = 'data'
    summary_directory = 'board'
    checkpoint_directory = 'trained'

    gpu_cnt = 1

    epoch = 200
    batch_size = 128

    learning_rate = 0.1
    momentum = 0.9
    wdecay = 5e-4

    pin_memory = True
    async_loading = True

    #############################################
    vae_batch_size = 1024
    vae_epoch = 2000

    vae_num_hiddens = 128
    vae_num_residual_hiddens = 32
    vae_num_residual_layers = 2

    vae_embedding_dim = 64
    vae_num_embeddings = 100

    vae_commitment_cost = 0.25

    vae_decay = 0.99

    vae_distance = 2.

    vae_learning_rate = 1e-3
