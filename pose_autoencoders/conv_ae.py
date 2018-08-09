import tensorflow as tf


def AutoEncoder(features, labels, mode):
    """
    Fully connected autoencoder
    :return:
    """
    # todo https://www.tensorflow.org/tutorials/estimators/cnn

    # batch size??
    input_layer = tf.reshape(features["x"], [-1, 48])

    encode1 = tf.layers.dense(input_layer, units=24, activation=tf.nn.relu)
    en_dropout1 = tf.layers.dropout(encode1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    encode2 = tf.layers.dense(en_dropout1, units=12, activation=tf.nn.relu)
    en_dropout2 = tf.layers.dropout(encode2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    encode3 = tf.layers.dense(en_dropout2, units=2, activation=tf.nn.relu)
    en_dropout3 = tf.layers.dropout(encode3, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    embedding = en_dropout3

    decoder1 = tf.layers.dense(embedding, units=12, activation=tf.nn.relu)
    de_dropout1 = tf.layers.dropout(decoder1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    decoder2 = tf.layers.dense(de_dropout1, units=24, activation=tf.nn.relu)
    de_dropout2 = tf.layers.dropout(decoder2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    decoder3 = tf.layers.dense(de_dropout2, units=48, activation=tf.nn.relu)
    de_dropout3 = tf.layers.dropout(decoder3, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    output_layer = de_dropout3  # todo needs to be separated / reimplemented in tf.js
