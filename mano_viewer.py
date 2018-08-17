"""
Saves posed hand models (.obj) based on the supplied poses (potentially originating from the scan registrations)

"""

import pickle

import numpy as np
import tensorflow as tf
from tf_smpl.batch_smpl import SMPL

print(tf.__version__)

with open('mpi/data/mano/MANO_RIGHT_py3.pkl', 'rb') as f:
    mano_data = pickle.load(f, encoding='latin1')

batch_size = mano_data['hands_coeffs'].shape[0]

with tf.Session() as sess:
    # init MANO
    mano = SMPL(mano_data)

    hands_components = tf.Variable(mano_data['hands_components'], dtype=tf.float32)
    hands_coeffs = tf.Variable(mano_data['hands_coeffs'], dtype=tf.float32)

    hands_poses = tf.matmul(hands_coeffs, hands_components)

    # cam = N x 3, pose N x self.num_theta, shape: N x 10
    rot = tf.Variable([[0, 3.14/2, 0]],expected_shape=[1,3])
    print(sess.run(tf.rank(rot)))
    cams = tf.tile(rot, [1554, 1])

    sess.run(tf.global_variables_initializer())
    print(sess.run(cams))

    poses = tf.Variable(tf.concat([cams, mano_data['hands_mean'] + hands_poses], axis=1))
    shapes = tf.Variable(tf.zeros([batch_size, 10]))

    # init vars
    sess.run(tf.global_variables_initializer())

    verts, joints, Rs = mano(shapes, poses, get_skin=True)

    verts = sess.run(verts)
for i, model in enumerate(verts):
    ## Write to an .obj file
    with open("./mano_objs/mano_{num}.obj".format(num=i), 'w') as fp:
        for v in model * 10.:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in mano_data['f'] + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

print('done')
