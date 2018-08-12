import tensorflow as tf
import tensorflow.contrib.slim as slim


def add_hidden_layer_summary(value):
    tf.summary.scalar('fraction_of_zero_values', tf.nn.zero_fraction(value))
    tf.summary.histogram('activation', value)

# def autoencoder_arg_scope(activation_fn, dropout, weight_decay, data_format, mode):
#     is_training = mode == tf.estimator.ModeKeys.TRAIN
#
#     if weight_decay is None or weight_decay <= 0:
#         weights_regularizer = None
#     else:
#         weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
#
#     with slim.arg_scope([tf.contrib.layers.fully_connected, conv2d_fixed_padding, tf.contrib.layers.conv2d_transpose],
#                         weights_initializer=slim.initializers.variance_scaling_initializer(),
#                         weights_regularizer=weights_regularizer,
#                         activation_fn=activation_fn),\
#          slim.arg_scope([slim.dropout],
#                         keep_prob=dropout,
#                         is_training=is_training),\
#          slim.arg_scope([conv2d_fixed_padding, tf.contrib.layers.conv2d_transpose],
#                         kernel_size=3, padding='SAME', data_format=data_format), \
#          slim.arg_scope([tf.contrib.layers.max_pool2d], kernel_size=2, data_format=data_format) as arg_sc:
#         return arg_sc