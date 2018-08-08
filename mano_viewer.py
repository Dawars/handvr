import pickle

import numpy as np
import tensorflow as tf
from tf_smpl.batch_smpl import SMPL

print(tf.__version__)

with open('mpi/data/mano/MANO_RIGHT_py3.pkl', 'rb') as f:
    mano_data = pickle.load(f, encoding='latin1')

with tf.Session() as sess:
    # init MANO
    mano = SMPL(mano_data)

    # cam = N x 3, pose N x self.num_theta, shape: N x 10
    # cams = theta_here[:, :self.num_cam]
    poses = tf.Variable(tf.zeros([1, 45 + 3]))
    shapes = tf.Variable(tf.zeros([1, 10]))

    # init vars
    sess.run(tf.global_variables_initializer())

    print(sess.run(mano.v_template))

    verts, joints, Rs = mano(shapes, poses, get_skin=True)

    verts = sess.run(verts)

## Write to an .obj file
outmesh_path = './MANO.obj'
with open(outmesh_path, 'w') as fp:
    for v in verts[0]:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

    for f in mano_data['f'] + 1:  # Faces are 1-based, not 0-based in obj files
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

print('done')
