import json
import tensorflow as tf
from tf_smpl.batch_smpl import SMPL
import pickle
import numpy as np


#with open('/datasets/handvr/mano/hands_components.json') as f:
#    hands_components = json.load(f)
#with open('/datasets/handvr/mano/hands_coeffs.json') as f:
#    hands_coeffs = json.load(f)


with open('mpi/data/mano/MANO_RIGHT_py3.pkl', 'rb') as f:
    mano_data = pickle.load(f, encoding='latin1')

mano = SMPL(mano_data)


def get_mano_vertices(shape, pose):
    vertices = mano(tf.Variable(shape), tf.Variable(pose), get_skin=True)
    return vertices[0]


def get_mano_faces():
    return mano_data['f']


def mano_to_OBJ(shape, pose, filename):
    model = get_mano_vertices(shape, pose)

    with open(filename, 'w') as fp:
        for v in model * 10.:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in mano_data['f'] + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
