"""
Rendering animation where the shape params are changing one by one
"""
import os

import numpy as np
from matplotlib import pyplot as plt

from utils.mano_utils import get_mano_vertices
from utils.render_manifold import HandRenderer

if __name__ == '__main__':
    save_path = './hand_shape_anim/outside'

    renderer = HandRenderer(image_size=512)

    iter = 32  # frame per component

    pose = np.zeros([1, 48])
    pose[:, :3] = [np.pi / -2, 0, 0]
    poses = np.repeat(pose.reshape(1, -1), iter, axis=0)

    sins = np.sin(np.arange(0, np.pi, np.pi / iter))

    for comp in range(10):
        shape_params = np.zeros([iter, 10])
        shape_params[:, comp] = sins / 10.

        verts = get_mano_vertices(pose=poses, shape=shape_params)
        for i, model in enumerate(verts):
            img = renderer.render_mano(model)
            # img.show()
            plt.imsave(os.path.join(save_path, f"hand_anim_{comp:02}_{i:04}.png"), img)
