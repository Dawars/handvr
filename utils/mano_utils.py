import os
import pickle

import numpy as np
import torch
from block_timer.timer import Timer

from smpl.batch_smpl import SMPL

with open('../mpi/data/mano/MANO_RIGHT_py3.pkl', 'rb') as f:
    mano_data = pickle.load(f, encoding='latin1')

mano = SMPL(mano_data).cuda()  # not good


def get_mano_vertices(shape, pose, device=torch.device('cpu')):
    """
    :param shape: mano shape params [batch_size, 10]
    :param pose: mano pose params including global rotation (joint axis-angle representation) [batch_size, 45+3]
    :return:
    """
    # check if not tensor: wrap
    if not isinstance(shape, torch.Tensor):
        shape = torch.tensor(shape, dtype=torch.float).cuda()

    if not isinstance(pose, torch.Tensor):
        pose = torch.tensor(pose, dtype=torch.float).cuda()

    verts, joints, Rs = mano(shape, pose, get_skin=True)
    return verts.cpu().detach().numpy()


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
    batch_size = 5

    # morph and skin
    with Timer():
        vertices = get_mano_vertices(np.zeros([batch_size, 10]), np.zeros([batch_size, 48]))
        print(vertices)

    # save obj
    save_mano_obj(vertices, './')

    # remap joints for physical proximity
    from pose_autoencoders.pose_loader import get_poses

    remapped = remap_joints(get_poses())

    print('done')
