import os
import argparse
import datetime
import numpy as np
import tensorflow as tf

from absl import logging

import data_loader
import model_builder
import loss_builder
import distribution_utils

logging.set_verbosity(logging.INFO)


def train(model, loss_fn, strategy, epochs, batch_size,
          train_dataset, validation_dataset, checkpoint_dir, log_dir):
    """Train models

    Args:
        model: Model to train
        loss_fn: Loss function
        strategy: Distribute strategy
        epochs: Num epochs
        batch_size: Batch size
        train_dataset: Dataset for train
        validation_dataset: Dataset for validation
        checkpoint_dir: Directory to output checkpoint
        log_dir: Directory to output TensorBoard logs

    Returns: Trained model
    """

    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam()
        train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
        val_acc = tf.keras.metrics.CategoricalAccuracy(name='val_acc')
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        model.compile(optimizer=optimizer, loss=loss_fn)
        start_epoch = 0

        # Train
        def replicated_train_step(inputs, y_true):
            with tf.GradientTape() as tape:
                y_pred = model(inputs)
                loss = loss_fn(y_true, y_pred)
                loss = tf.reduce_sum(loss, keepdims=True) * (1.0 / batch_size)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_acc.update_state(y_true, y_pred)

            return loss

        @tf.function
        def train_step(inputs, y_true):
            per_replica_losses = strategy.experimental_run_v2(
                replicated_train_step, args=(inputs, y_true))
            loss = strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            return loss

        # Validation
        def replicated_validation_step(inputs, y_true):
            y_pred = model(inputs)
            loss = loss_fn(y_true, y_pred)
            loss = tf.reduce_sum(loss, keepdims=True) * (1.0 / batch_size)
            val_acc.update_state(y_true, y_pred)

            return loss

        @tf.function
        def validation_step(inputs, y_true):
            per_replica_losses = strategy.experimental_run_v2(
                replicated_validation_step, args=(inputs, y_true))
            loss = strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            return loss

        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        tb_callback.set_model(model)

        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoints = os.listdir(checkpoint_dir)
        if len(checkpoints) > 0:
            ckpt = checkpoints[-1]
            model = tf.keras.models.load_model(
                os.path.join(checkpoint_dir, ckpt))
            optimizer = model.optimizer
            start_epoch = optimizer.iterations.numpy() // batch_size
            logging.info(f'load checkpoint: {ckpt}')
            logging.info(f'start steps: {optimizer.iterations.numpy()}')
            logging.info(f'start epoch: {start_epoch}')

        summary_writer = tf.summary.create_file_writer(log_dir)
        best_val_loss = np.inf
        labels = tf.range(1, 11, dtype=tf.float32)
        for epoch in range(start_epoch, epochs):
            # Train
            for inputs, y_true in train_dataset:
                loss = train_step(inputs, y_true)
                train_loss.update_state(loss)

                with summary_writer.as_default():
                    tf.summary.scalar('train_loss', train_loss.result(),
                                      step=optimizer.iterations)
                    tf.summary.scalar('train_acc', train_acc.result(),
                                      step=optimizer.iterations)

                global_step = optimizer.iterations.numpy()
                if (global_step + 1) % 50 == 0:
                    current_time = datetime.datetime.now().strftime(
                        '%Y/%m/%d-%H:%M:%S')
                    logging.info(
                        f'[{current_time}] '
                        f'epoch: {epoch}, global_step: {global_step} '
                        f'train_loss: {train_loss.result():0.4f}, '
                        f'train_acc: {train_acc.result():0.4f}')
                    train_acc.reset_states()
                    train_loss.reset_states()

            # Validation
            for inputs, y_true in validation_dataset:
                loss = validation_step(inputs, y_true)
                val_loss.update_state(loss)

            with summary_writer.as_default():
                tf.summary.scalar('val_loss', val_loss.result(),
                                  step=optimizer.iterations)
                tf.summary.scalar('val_acc', val_acc.result(),
                                  step=optimizer.iterations)

            current_time = datetime.datetime.now().strftime('%Y/%m/%d-%H:%M:%S')
            global_step = optimizer.iterations.numpy()
            logging.info(f'[{current_time}] '
                         f'epoch: {epoch}, global_step: {global_step} '
                         f'val_loss: {val_loss.result():0.4f}, '
                         f'val_acc: {val_acc.result():0.4f}')

            if val_loss.result() < best_val_loss:
                export_path = os.path.join(
                    checkpoint_dir,
                    f'ckpt_{epoch}_{optimizer.iterations.numpy()}_'
                    f'{val_loss.result():0.4f}_{val_acc.result():0.4f}.h5')
                model.save(export_path)
                logging.info(
                    f'Save model to {export_path} due to update best val loss '
                    f'{val_loss.result():0.4f} < {best_val_loss:0.4f}')
                best_val_loss = val_loss.result()

            val_acc.reset_states()
            val_loss.reset_states()

        return model


def run(params):
    """Run training

    Args:
        params: Parameters for training

    Returns: None

    """

    logging.info(f'params: {params}')
    strategy = distribution_utils.get_distribution_strategy(
        params.get('tpu_address'))
    batch_size = distribution_utils.update_batch_size(strategy,
                                                      params['batch_size'])

    with strategy.scope():
        model = model_builder.build_model()
    input_image_size = model.input_shape[1]

    # Build dataset
    train_image_paths, train_scores = data_loader.read_csv(
        params['train_csv'], params['image_dir'], is_training=True)
    validation_image_paths, validation_scores = data_loader.read_csv(
        params['validation_csv'], params['image_dir'], is_training=False)


    train_dataset = data_loader.build_dataset(
        train_image_paths, train_scores,
        is_training=True, batch_size=batch_size, target_size=input_image_size)

    validation_dataset = data_loader.build_dataset(
        validation_image_paths, validation_scores,
        is_training=False, batch_size=batch_size, target_size=input_image_size)

    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    validation_dataset = strategy.experimental_distribute_dataset(
        validation_dataset)

    loss_fn = loss_builder.build_loss_fn(
        loss_name=params['loss'], trainable_variables=model.trainable_variables)

    train(model=model, loss_fn=loss_fn, strategy=strategy,
          epochs=params['epochs'], batch_size=batch_size,
          train_dataset=train_dataset, validation_dataset=validation_dataset,
          checkpoint_dir=params['checkpoint_dir'], log_dir=params['log_dir'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu_address', type=str,
                        help='TPU gRPC address. In Colab, we can get from os '
                             'environment value like `grpc://${COLAB_TPU_ADDR}`.')
    parser.add_argument('--train_csv', type=str, required=True,
                        help='/path/to/AVA_train.csv. Allow GCS address like '
                             '`gs://<bucket_name>/AVA_train.csv`.')
    parser.add_argument('--validation_csv', type=str, required=True,
                        help='/path/to/AVA_validation.csv. Allow GCS address like '
                             '`gs://<bucket_name>/AVA_validation.csv`.')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='/path/to/images. Allow GCS address like '
                             '`gs://<bucket_name>/images`.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='The number of epochs for train.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size in training and validation.')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='/path/to/checkpoints. Checkpoints will be stored '
                             'to here. Checkpoints dir does NOT support GCS address.')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='/path/to/logs. TensorBoard log will be stored '
                             'to here. Allow GCS address like '
                             '`gs://<bucket_name>/logs`.')
    parser.add_argument('--loss', type=str, default='multi_loss',
                        help='Name of loss function.')

    args = parser.parse_args()

    run(vars(args))
