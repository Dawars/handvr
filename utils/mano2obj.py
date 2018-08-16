import json
import tensorflow as tf
from tf_smpl.batch_smpl import SMPL
import pickle


#with open('/datasets/handvr/mano/hands_components.json') as f:
#    hands_components = json.load(f)
#with open('/datasets/handvr/mano/hands_coeffs.json') as f:
#    hands_coeffs = json.load(f)


with open('mpi/data/mano/MANO_RIGHT_py3.pkl', 'rb') as f:
    mano_data = pickle.load(f, encoding='latin1')

mano = SMPL('/projects/handvr/mpi/data/mano/mano_params.pkl')


def manoToOBJ(shapes, poses, filename):
    vertices = mano(shapes, poses, get_skin=True)
    model = vertices[0]

    with open(filename, 'w') as fp:
        for v in model * 10.:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in mano_data['f'] + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

