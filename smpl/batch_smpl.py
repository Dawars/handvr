""" 
Tensorflow SMPL implementation as batch.
Note: To get original smpl joints, use self.J_transformed

source: https://github.com/akanazawa/hmr
Modified to work with Mano in Python3 by @Dawars
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle

import torch
from block_timer.timer import Timer

from smpl.batch_lbs import batch_rodrigues, batch_global_rigid_transformation


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


class SMPL(torch.nn.Module):
    def __init__(self, dd, dtype=torch.float):
        """
        pkl_path is the path to a SMPL model
        """
        super(SMPL, self).__init__()
        self.register_buffer('v_template', torch.tensor(
            undo_chumpy(dd['v_template']),
            dtype=dtype,
            requires_grad=False))
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis: 778 x 3 x 10
        # reshaped to 778*30 x 10, transposed to 10x778*3
        shapedir = np.reshape(undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.tensor(shapedir, dtype=dtype, requires_grad=False))

        # Regressor for joint locations given shape - 778 x 16
        self.register_buffer('J_regressor',
                             torch.tensor(dd['J_regressor'].T.todense(),
                                          dtype=dtype,
                                          requires_grad=False))

        # Pose blend shape basis: 778 x 3 x 135, reshaped to 778*3 x 135
        num_pose_basis = dd['posedirs'].shape[-1]
        # 135 x 2334
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.tensor(
            posedirs, dtype=dtype, requires_grad=False))

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.register_buffer('weights', torch.tensor(
            undo_chumpy(dd['weights']),
            dtype=dtype,
            requires_grad=False))

        # This returns 15 keypoints: 778 x 16
        self.register_buffer('joint_regressor', torch.tensor(
            dd['J_regressor'].T.todense(),
            dtype=dtype,
            requires_grad=False))

        self.register_buffer('e3', torch.eye(3).float())

        self.cur_device = None

    def forward(self, beta, theta, get_skin=False):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 48 (with 3-D axis-angle rep) [float]

        Updates:
        self.J_transformed: N x 16 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 16 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 778 x 3
        """

        if not self.cur_device:
            device = beta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 16, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 135)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base=True)

        W = self.weights.repeat(num_batch, 1).view(num_batch, -1, 16)
        T = torch.matmul(W, A.view(num_batch, 16, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints


if __name__ == '__main__':
    with open('../mpi/data/mano/MANO_RIGHT_py3.pkl', 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')

    device = torch.device('cuda', 0)

    # init MANO
    mano = SMPL(mano_data).to(device)
    with Timer():

        batch_size = 1554

        # processing `batch_size` models at once
        rot = torch.tensor([[0, 3.14 / 2, 0]])  # global rotation
        cams = rot.view(1, 3).repeat(batch_size, 1).view(batch_size, 3)

        hands_poses = np.random.uniform(-np.pi / 2, np.pi / 2, [batch_size, 45])

        poses = torch.tensor(np.concatenate([cams, mano_data['hands_mean'] + hands_poses], axis=1),
                             dtype=torch.float).to(device)
        shapes = torch.zeros([batch_size, 10]).to(device)

        for i in range(1):
            verts, joints, Rs = mano(shapes, poses, get_skin=True)
            model_verts = verts.cpu()

    print(model_verts.shape)
    # print(model_verts)
