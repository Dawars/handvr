"""
In this notebook I will compare different non-linear Auto Encoder based methods to reduce the dimensionality of the hand
 pose space to 2D
 This is desirable for easy exploration of the shape space. This enables rendering a manifold of the space
    - Fully Connected AE
    - Convolutional AE
    - Fully Connected VAE
    - Convolutional VAE
"""

import pickle

import numpy as np
import tensorflow as tf
from tf_smpl.batch_smpl import SMPL

print(tf.__version__)


def get_poses():
    """
    Return a Tensor containing the poses
    (1554, 45) - 3*15 = 45 joint anles for 1554 people
    """
    with open('mpi/data/mano/MANO_RIGHT_py3.pkl', 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')

    batch_size = mano_data['hands_coeffs'].shape[0]

    hands_components = tf.Variable(mano_data['hands_coeffs'], dtype=tf.float32)
    hands_coeffs = tf.Variable(mano_data['hands_components'], dtype=tf.float32)

    # 3*15 = 45 joint angles
    hands_poses = tf.matmul(hands_components, hands_coeffs)

    return hands_poses


def main():
    with tf.Session() as sess:
        poses = get_poses()

        sess.run(tf.global_variables_initializer())
        print(poses.eval().shape)


if __name__ == '__main__':
    main()
