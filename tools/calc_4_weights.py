"""
Calculate the 4 most influential bones affecting each vertex
"""
import json
import numpy as np


def main():
    weights = json.load(open('/Users/dawars/projects/handvr/MANO/mano_params/weights.json'))

    indices = np.argsort(weights, axis=1)[:, 12:]#len-4?
    values = np.sort(weights, axis=1)[:, 12:]

    with open('/Users/dawars/projects/HandVR/MANO/mano_params/weights_indices.json', 'w') as f:
        f.write(json.dumps(indices.tolist()))
    with open('/Users/dawars/projects/HandVR/MANO/mano_params/weights4.json', 'w') as f:
        f.write(json.dumps(values.tolist()))

    pass


if __name__ == '__main__':
    main()
