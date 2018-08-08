import json
import tensorflow as tf
from hmr.src.tf_smpl.batch_smpl import SMPL

with open('/datasets/handvr/mano/hands_components.json') as f:
    hands_components = json.load(f)
with open('/datasets/handvr/mano/hands_coeffs.json') as f:
    hands_coeffs = json.load(f)

mano = SMPL('/projects/handvr/mpi/data/mano/mano_params.pkl')


print('end')
