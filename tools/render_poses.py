import numpy as np

from pose_autoencoders.pose_loader import get_poses
from utils.render_manifold import HandRenderer, mano_data, get_mano_vertices

if __name__ == '__main__':
    renderer = HandRenderer(image_size=128, num_vertices=778)

    # low poly
    poses = get_poses() + mano_data['hands_mean']

    rot = np.zeros([16 * 16, 3])
    rot[:, 0] = np.pi / 4

    poses = np.concatenate((rot, poses[:16 * 16]), axis=1)
    vertices = get_mano_vertices(pose=poses, shape=np.zeros([16 * 16, 10]))

    renderer.render_hands(vertices, dims=(16, 16), filename='./mano_poses_low.png')
