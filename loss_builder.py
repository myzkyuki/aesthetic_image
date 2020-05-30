import re
import tensorflow as tf


class MultiLoss(tf.keras.losses.Loss):
    def __init__(self, labels, trainable_variables, reduction,
                 alpha=1.0, beta=1e-5):
        super(MultiLoss, self).__init__(reduction=reduction)
        self.labels = labels
        self.sce = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=reduction)
        self.mse = tf.keras.losses.MSE
        self.trainable_variables = [v for v in trainable_variables
                                    if re.match(r'.*(kernel|weight):0$',
                                                v.name)]
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        y_true_mean = tf.reduce_sum(y_true * self.labels, axis=1)
        y_true_bucketized = tf.math.round(y_true_mean) - 1
        loss_sce = self.sce(y_true_bucketized, y_pred)

        y_pred_mean = tf.reduce_sum(y_pred * self.labels, axis=1)
        loss_mse = self.mse(y_true_mean, y_pred_mean)
        loss = loss_sce + self.alpha * loss_mse

        # Weight decay loss
        loss += self.beta * tf.add_n([tf.nn.l2_loss(v)
                                      for v in self.trainable_variables])

        return loss


def build_loss_fn(loss_name, trainable_variables):
    if loss_name == 'multi_loss':
        labels = tf.range(1, 11, dtype=tf.float32)
        loss_fn = MultiLoss(labels, trainable_variables,
                            reduction=tf.keras.losses.Reduction.NONE)
    elif loss_name == 'kldivergence':
        loss_fn = tf.keras.losses.KLDivergence(
            reduction=tf.keras.losses.Reduction.NONE)
    else:
        raise ValueError(f'Unknown loss name {loss_name}')

    return loss_fn
