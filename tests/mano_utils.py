import unittest
import numpy as np

from utils.mano_utils import *


class ManoTests(unittest.TestCase):
    def test_tensor_inputs(self):
        batch_size = 5

        # morph and skin
        shape = torch.tensor(np.zeros([batch_size, 10]), dtype=torch.float)
        poses = torch.tensor(np.zeros([batch_size, 48]), dtype=torch.float)
        vertices = get_mano_vertices(shape, poses, torch.device('cpu'))
        vertices.backwards()
        self.assertEqual()

    def test_verts_cpu(self):
        batch_size = 5

        # morph and skin
        vertices = get_mano_vertices(np.zeros([batch_size, 10]), np.zeros([batch_size, 48]), torch.device('cpu'))
        self.assertEqual(vertices.device(), 'cpu', "Tensor on wrong device")

    def test_verts_cuda(self):
        batch_size = 5

        # morph and skin
        vertices = get_mano_vertices(np.zeros([batch_size, 10]), np.zeros([batch_size, 48]), torch.device('cuda', 0))
        self.assertEqual(vertices.device().type, 'cuda', "Tensor on wrong device")

    def test_save_obj(self):
        batch_size = 1
        vertices = get_mano_vertices(np.zeros([batch_size, 10]), np.zeros([batch_size, 48]))

        # save obj
        save_mano_obj(vertices, './')

        self.assertEqual(os.path.isfile('./mano_0.obj'), True, "Obj file not found")

    def test_remap_joint(self):
        # remap joints for physical proximity
        from pose_autoencoders.pose_loader import get_poses

        remapped = remap_joints(get_poses())


if __name__ == '__main__':
    unittest.main()
