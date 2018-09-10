from plyfile import PlyData

from pose_autoencoders.pose_loader import get_poses
from utils.render_manifold import HandRenderer, mano_data, get_mano_vertices

import numpy as np

if __name__ == '__main__':
    renderer = HandRenderer(image_size=128, num_vertices=778)

    # low poly
    poses = get_poses() + mano_data['hands_mean']

    rot = np.zeros([16 * 16, 3])
    rot[:, 0] = np.pi / 4

    poses = np.concatenate((rot, poses[:16 * 16]), axis=1)
    # vertices = get_mano_vertices(pose=poses, shape=np.zeros([16 * 16, 10]))

    # renderer.render_hands(vertices, dims=(16, 16), filename='./mano_poses_low.png')

    # high poly
    renderer = HandRenderer(image_size=128, num_vertices=39003)
    # todo faces

    plydata = PlyData.read('/home/dawars/Downloads/handsOnly_SCANS/01_01r.ply')['vertex']

    verts = np.stack((plydata['x'], plydata['y'], plydata['z']), axis=-1)
    print(verts)

    img = renderer.render_mano(verts)
    img.save('./mano_hand_high.png')
