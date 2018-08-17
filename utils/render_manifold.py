"""
Functions for rendering a single MANO model to image and manifold
"""

import numpy as np
import trimesh
from autolab_core import RigidTransform
from perception import CameraIntrinsics

from meshrender import Scene, MaterialProperties, AmbientLight, PointLight, SceneObject, VirtualCamera

import torch
from pose_autoencoders.pytorch_ae.vanilla_ae import autoencoder
from utils.vertex_utils import get_mano_vertices, get_mano_faces, mano_to_OBJ
from PIL import Image


# Load a premade autoencoder
ae = autoencoder()
ae.load_state_dict(torch.load('pose_autoencoders/pytorch_ae/sim_autoencoder.pth'))


# Settings

render_w = 500
render_h = 500
# render_aspect = render_w / render_h

image_w = 200  # Height of individual images
image_h = 200  # The actual height may be different to keep the original aspect ratio

step_x = 1
step_y = 1

grid_x_min = -6  # Dimensions of the space to render
grid_x_max = 6
grid_y_min = -6
grid_y_max = 6

result_w = image_w * (grid_x_max - grid_x_min)
result_h = image_h * (grid_y_max - grid_y_min)


def render_manifold():
    shape = [[0] * 10]
    res = Image.new("RGB", (result_w, result_h))
    for x in range(grid_x_min, grid_x_max, step_x):
        for y in range(grid_y_min, grid_y_max, step_y):
            print("Rendering at {x}, {y}".format(x=x, y=y))

            encoded = torch.tensor([x, y], dtype=torch.float)
            decoded_pose = ae.decoder(encoded).cpu().data.numpy()
            decoded_pose = np.concatenate(([0, 0, 0], decoded_pose))
            vertices = get_mano_vertices(shape, decoded_pose)
            mesh = trimesh.Trimesh(vertices=vertices, faces=get_mano_faces(), process=False)

            raw = render_mano(mesh)[0]
            img = Image.fromarray(raw, "RGB")
            img.thumbnail((200, 200), Image.ANTIALIAS)
            # mano_to_OBJ(shape, decoded_pose, "./test.obj")
            res.paste(img, ((x - grid_x_min) * image_w, (y - grid_y_min) * image_h))
    print("Images rendered")
    res.save("./manifold.png")


def render_mano(mesh):
    """
    Render Mano on a single image
    :param mesh:
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
        cx=319.5,
        cy=239.5,
        skew=0.0,
        height=render_h,
        width=render_w
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


render_manifold()