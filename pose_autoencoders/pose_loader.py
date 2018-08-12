"""
The input function for the Estimator
"""

import pickle

import numpy as np
import tensorflow as tf

# tf.flags.DEFINE_float("dropout", 0.8, "Dropout after the bottleneck layer")
# tf.flags.DEFINE_float("learning_rate", 0.005, "Learning rate")
# tf.flags.DEFINE_integer("batch_size", 32, "Batch size")
#
# tf.flags.DEFINE_string("logdir", './model_files/', "Batch size")

FLAGS = tf.flags.FLAGS


def get_poses():
    """
    Return a Tensor containing the poses
    (1554, 45) - 3*15 = 45 joint anles for 1554 people
    """
    with open('../mpi/data/mano/MANO_RIGHT_py3.pkl', 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')

    batch_size = mano_data['hands_coeffs'].shape[0]

    hands_components = mano_data['hands_coeffs']
    hands_coeffs = mano_data['hands_components']

    # 3*15 = 45 joint angles
    hands_poses = np.matmul(hands_components, hands_coeffs)

    print(FLAGS.batch_size)
    return hands_poses


def pose_input_fn():
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=get_poses(),
        batch_size=FLAGS.batch_size,
        num_epochs=None,
        shuffle=True,
        queue_capacity=1554,
        num_threads=1,
    )
    return train_input_fn


def main(argv):
    with tf.Session() as sess:
        poses = get_poses()
        print(poses.shape)
        print(poses.max())
        print(poses.min())


if __name__ == '__main__':
    tf.app.run(main)
