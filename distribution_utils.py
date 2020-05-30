import tensorflow as tf
from absl import logging

logging.set_verbosity(logging.INFO)


def get_distribution_strategy(tpu_address):
    """Get distribution strategy

    Args:
        tpu_address: TPU gRPC address

    Returns: Distribution strategy

    """
    # Detect hardware
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)  # TPU detection
    except (ValueError, KeyError):
        tpu = None
        gpus = tf.config.experimental.list_logical_devices("GPU")

    # Select appropriate distribution strategy
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        logging.info(f'Running on TPU {tpu.cluster_spec().as_dict()["worker"]}')
    elif len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        logging.info(f'Running on multiple GPUs {[gpu.name for gpu in gpus]}')
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()
        logging.info(f'Running on single GPU {gpus[0].name}')
    else:
        strategy = tf.distribute.get_strategy()
        logging.info('Running on CPU')
    logging.info(f'Number of accelerators: {strategy.num_replicas_in_sync}')

    return strategy


def update_batch_size(strategy, batch_size):
    num_replicas = strategy.num_replicas_in_sync
    if batch_size % num_replicas != 0:
        logging.warning(f'The batch size ({batch_size}) must be a multiple of '
                        f'the number of replicas ({num_replicas}).')

        batch_size = batch_size // num_replicas * num_replicas
        logging.info(f'Update batch size to {batch_size}.')

    return batch_size
