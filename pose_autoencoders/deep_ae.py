import tensorflow as tf
from tensorflow.contrib import slim

#
# class AutoEncoder(tf.estimator.Estimator):
#     """An Autoencoder estimator with fully connected layers.
#     Parameters
#     ----------
#     hidden_units : list of int
#         Number of units in each hidden layer.
#     activation_fn : callable|None
#         Activation function to use.
#     dropout : float|None
#          Percentage of nodes to remain activate in each layer,
#          or `None` to disable dropout.
#     weight_decay : float|None
#         Amount of regularization to use on the weights
#         (excludes biases).
#     learning_rate : float
#         Learning rate.
#     model_dir : str
#         Directory where outputs (checkpoints, event files, etc.)
#         are written to.
#     config : RunConfig
#         Information about the execution environment.
#     """

def deep_ae_model_fn(features, labels, mode):
        # Define model's architecture
        net = features
        net = fc_encoder(net, hidden_units)

def fc_encoder(inputs, hidden_units, dropout, scope=None):
    net = inputs
    with tf.variable_scope(scope, 'encoder', [inputs]):
        tf.assert_rank(inputs, 2)
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope(
                    'layer_{}'.format(layer_id),
                    values=(net,)) as layer_scope:
                net = tf.contrib.layers.fully_connected(
                    net,
                    num_outputs=num_hidden_units,
                    scope=layer_scope)
                if dropout is not None:
                    net = slim.dropout(net)
                self.add_hidden_layer_summary(net)
        net = tf.identity(net, name='output')

    return net

def fc_decoder(inputs, hidden_units, dropout, scope=None):
    net = inputs
    with tf.variable_scope(scope, 'decoder', [inputs]):
        for layer_id, num_hidden_units in enumerate(hidden_units[:-1]):
            with tf.variable_scope(
                    'layer_{}'.format(layer_id),
                    values=(net,)) as layer_scope:
                net = tf.contrib.layers.fully_connected(
                    net,
                    num_outputs=num_hidden_units,
                    scope=layer_scope)
                if dropout is not None:
                    net = slim.dropout(net, scope=layer_scope)
                self.add_hidden_layer_summary(net)

        with tf.variable_scope(
                'layer_{}'.format(len(hidden_units) - 1),
                values=(net,)) as layer_scope:
            net = tf.contrib.layers.fully_connected(net, hidden_units[-1],
                                                    activation_fn=None,
                                                    scope=layer_scope)
            tf.summary.histogram('activation', net)
        net = tf.identity(net, name='output')
    return net
