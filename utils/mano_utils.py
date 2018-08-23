import os
import pickle

import numpy as np
import tensorflow as tf

from tf_smpl.batch_smpl import SMPL

with open('../mpi/data/mano/MANO_RIGHT_py3.pkl', 'rb') as f:
    mano_data = pickle.load(f, encoding='latin1')

mano = SMPL(mano_data)


def get_mano_vertices(shape, pose, sess=tf.Session()):
    """

    :param sess: TensorFlow session to run with
    :param shape: mano shape params [batch_size, 10]
    :param pose: mano pose params including global rotation (joint axis-angle representation) [batch_size, 45+3]
    :return:
    """
    verts, joints, Rs = mano(tf.Variable(shape, dtype=tf.float32), tf.Variable(pose, dtype=tf.float32), get_skin=True)
    sess.run(tf.global_variables_initializer())
    return sess.run(verts)


def get_mano_faces():
    return mano_data['f']


def save_mano_obj(model, save_path):
    """
    Writes mano model to and OBJ file
    :param model: vertices list of the mano model or array of mano models
    :param save_path: path to save to
    :return:
    """
    os.makedirs(save_path, exist_ok=True)

    model = np.reshape(model, [-1, 778, 3])  # reshape single model to a list
    for i, model in enumerate(model):
        # Write to an .obj file
        with open(os.path.join(save_path, "mano_{num}.obj".format(num=i)), 'w') as fp:
            for v in model * 10.:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in mano_data['f'] + 1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def remap_joints(joints):
    """
    Remap joint orientation params to follow the physical placement (thumb, index, middle, ring, pinky)
    The order of joints for mano is: index, middle, pinky, ring, thumb
    :param joints: array with joint rotations in mano format
    :return: remapped array [batch_size, 5, 3x3]
    """

    indices = [
        np.arange(4 * 9, 5 * 9, 1),
        np.arange(0 * 9, 1 * 9, 1),
        np.arange(1 * 9, 2 * 9, 1),
        np.arange(3 * 9, 4 * 9, 1),
        np.arange(2 * 9, 3 * 9, 1),
    ]

    return joints[:, indices]


if __name__ == '__main__':
    batch_size = 2

    with tf.Session() as sess:
        vertices = get_mano_vertices(np.zeros([batch_size, 10]), np.zeros([batch_size, 48]), sess)
        print(vertices)

    remapped = remap_joints(get_poses())

    print('done')
