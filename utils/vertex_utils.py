import json
import tensorflow as tf
from tf_smpl.batch_smpl import SMPL
import pickle
import numpy as np


#with open('/datasets/handvr/mano/hands_components.json') as f:
#    hands_components = json.load(f)
#with open('/datasets/handvr/mano/hands_coeffs.json') as f:
#    hands_coeffs = json.load(f)

sess = tf.Session()

with open('mpi/data/mano/MANO_RIGHT_py3.pkl', 'rb') as f:
    mano_data = pickle.load(f, encoding='latin1')

mano = SMPL(mano_data)

hands_components = tf.Variable(mano_data['hands_components'], dtype=tf.float32)
hands_coeffs = tf.Variable(mano_data['hands_coeffs'], dtype=tf.float32)


def get_mano_vertices(shape, pose):
    vertices = mano(tf.Variable(shape, dtype=tf.float32), tf.Variable(pose, dtype=tf.float32), get_skin=True)
    sess.run(tf.global_variables_initializer())
    return sess.run(vertices)[0][0]


def get_mano_faces():
    return mano_data['f']


def mano_to_OBJ(shape, pose, filename):
    model = get_mano_vertices(shape, pose)

    with open(filename, 'w') as fp:
        for v in model * 10.:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in mano_data['f'] + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
