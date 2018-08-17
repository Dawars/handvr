"""
Functions for rendering a single MANO model to image and manifold
"""

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch

import trimesh
from autolab_core import RigidTransform
from perception import CameraIntrinsics
from meshrender import Scene, MaterialProperties, AmbientLight, PointLight, SceneObject, VirtualCamera

from pose_autoencoders.vanilla_ae import autoencoder
from utils.mano_utils import *


def render_manifold(decoder, name="./manifold.png", bounds=(-4, 4), steps=0.5, image_size=200, verbose=False):
    """
    Render a 2D posed hand manifold
    :param decoder: pytorch decoder function 2 -> 45 params
    :param name: filename
    :param bounds: bounds of the sampling along the x and y axis
    :param steps: step size for sampling between the bounds
    :param image_size: side length of a single hand in the manifold
    :param verbose: print progress
    :returns rendered image
    """
    # Settings
    supersampling = 2.5  # don't change
    result_length = image_size * (bounds[1] - bounds[0]) / steps

    # coordinates to sample at
    sampling_grid = np.mgrid[bounds[0]:bounds[1]:steps, bounds[0]:bounds[1]:steps]
    encoded = sampling_grid.reshape(2, -1).T

    _, cols, rows = sampling_grid.shape

    encoded = torch.tensor(encoded, dtype=torch.float)
    batch_size = len(encoded)

    rot = np.zeros([batch_size, 3])
    shape = np.zeros([batch_size, 10])

    decoded_poses = decoder(encoded).cpu().data.numpy()

    decoded_poses = np.concatenate((rot, decoded_poses), axis=1)
    vertices = get_mano_vertices(shape, decoded_poses)

    res = Image.new("RGB", (int(result_length), int(result_length)))
    for x in range(cols):
        for y in range(rows):
            if verbose:
                print("Rendering at {x}, {y}".format(x=x, y=y))

            model_index = y * rows + x
            model_verts = vertices[model_index]
            mesh = trimesh.Trimesh(vertices=model_verts, faces=get_mano_faces(), process=False)

            raw = render_mano(mesh, int(image_size * supersampling))[0]
            img = Image.fromarray(raw, "RGB")
            img.thumbnail((200, 200), Image.ANTIALIAS)
            # mano_to_OBJ(shape, decoded_poses, "./test.obj")

            x_pos = x * image_size
            y_pose = y * image_size

            res.paste(img, (int(x_pos), int(y_pose)))
    if verbose:
        print("Images rendered")
    res.save(name)
    return res


def render_mano(mesh, image_size):
    """
    Render Mano on a single image
    :param mesh: mesh to render
    :param image_size: size of rendered image
    :return tuple of (image_h, image_w, 3) with rgb values as ints between 0 and 255:
    """
    # Start with an empty scene
    scene = Scene(background_color=np.array([0, 0, 0]))

    # Set up pose in the world
    pose = RigidTransform(
        rotation=np.array(
            [[0, -1, 0],
             [1, 0, 0],
             [-0, 0, 1]]
        ),
        translation=np.array([0.0, 0.0, 0.0]),
        from_frame='obj',
        to_frame='world'
    )

    # Set up each object's material properties
    cube_material = MaterialProperties(
        color=np.array([1, 0.1, 0.2]),
        k_a=0.3,
        k_d=1.0,
        k_s=1.0,
        alpha=10.0,
        smooth=True
    )

    # Create SceneObjects for each object
    hand_obj = SceneObject(mesh, pose, cube_material)

    scene.add_object('hand', hand_obj)

    # Create an ambient light
    ambient = AmbientLight(
        color=np.array([1.0, 1.0, 1.0]),
        strength=0.5
    )

    # Add the lights to the scene
    scene.ambient_light = ambient  # only one ambient light per scene

    point = PointLight(
        location=np.array([1.0, 2.0, 3.0]),
        color=np.array([1.0, 1.0, 1.0]),
        strength=10.0
    )
    scene.add_light('point_light_one', point)

    # Set up camera intrinsics
    ci = CameraIntrinsics(
        frame='camera',
        fx=525.0,
        fy=525.0,
        cx=250,
        cy=250,
        skew=0.0,
        height=image_size,
        width=image_size
    )

    # Set up the camera pose (z axis faces away from scene, x to right, y up)
    cp = RigidTransform(
        rotation=np.array([
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0]
        ]),
        translation=np.array([0.3, 0.0, 0.0]),
        from_frame='camera',
        to_frame='world'
    )

    # Create a VirtualCamera
    camera = VirtualCamera(ci, cp)

    # Add the camera to the scene
    scene.camera = camera

    img = scene.render(render_color=True)
    return img


if __name__ == '__main__':
    # rendering mano
    mesh = trimesh.Trimesh(vertices=get_mano_vertices(np.zeros([1, 10]), np.zeros([1, 48]))[0],
                           faces=get_mano_faces(), process=False)
    img = render_mano(mesh, 500)
    plt.imsave('rendering_test.png', img[0])

    # rendering manifold
    ae = autoencoder()  # Load a premade autoencoder
    ae.load_state_dict(torch.load('../pose_autoencoders/sim_autoencoder.pth'))

    render_manifold(ae.decoder, 'manifold_test.png', verbose=True)
