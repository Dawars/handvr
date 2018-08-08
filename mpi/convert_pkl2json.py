import json
import os
import pickle as pkl

# every_gender = ['male', 'female', 'generic']
# for gender in every_gender:
with open('/projects/handvr/mpi/data/mano/MANO_RIGHT.pkl') as f:
    model = pkl.load(f)

    # posedirs = model['posedirs']

    # shapedirs = model['shapedirs'].x * 10.

    # shapes = []

    # for i in range(shapedirs.shape[2]):
    #     morphtarget = shapedirs[:, :, i].tolist()
    #     shapes.append(morphtarget)

    mat = {
        'hands_components': model['hands_components'].tolist(),
        'hands_coeffs': model['hands_coeffs'].tolist(),
        # 'J': model['J'].tolist(),
        # 'J_regressor': model['J_regressor'].todense().tolist(),
        # 'f': model['f'].tolist(),
        # 'shapes': shapes,
        # 'kintree_table': model['kintree_table'].tolist(),
        # 'v_template': (model['v_template']*100.).tolist(),
        # 'vert_sym_idxs': model['vert_sym_idxs'].tolist(),
        # 'weights': model['weights'].tolist(),
    }
    for key in mat.keys():
        with open(os.path.join('/projects/handvr/mpi/data/mano/', key + '.json'), 'w') as out:
            out.write(json.dumps(mat[key]))
