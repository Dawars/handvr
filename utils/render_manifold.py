"""
Functions for rendering a single MANO model to image and manifold
"""

import numpy as np
import trimesh
from autolab_core import RigidTransform
from perception import CameraIntrinsics, RenderMode

from meshrender import Scene, MaterialProperties, AmbientLight, PointLight, SceneObject, VirtualCamera


def render_mano(mesh):
    """
    Render Mano on a single image
    :param mesh:
    :return:
    """
    # Start with an empty scene
    scene = Scene()

    # Set up pose in the world
    pose = RigidTransform(
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 0.0]),
        from_frame='obj',
        to_frame='world'
    )

    # Set up each object's material properties
    cube_material = MaterialProperties(
        color=np.array([0.1, 0.1, 0.5]),
        k_a=0.3,
        k_d=1.0,
        k_s=1.0,
        alpha=10.0,
        smooth=False
    )

    # Create SceneObjects for each object
    hand_obj = SceneObject(mesh, pose, cube_material)

    scene.add_object('hand', hand_obj)

    # Create an ambient light
    ambient = AmbientLight(
        color=np.array([1.0, 1.0, 1.0]),
        strength=1.0
    )

    # Add the lights to the scene
    scene.ambient_light = ambient  # only one ambient light per scene

    # Set up camera intrinsics
    ci = CameraIntrinsics(
        frame='camera',
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
        skew=0.0,
        height=480,
        width=640
    )

    # Set up the camera pose (z axis faces away from scene, x to right, y up)
    cp = RigidTransform(
        rotation=np.array([
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0]
        ]),
        translation=np.array([-0.3, 0.0, 0.0]),
        from_frame='camera',
        to_frame='world'
    )

    # Create a VirtualCamera
    camera = VirtualCamera(ci, cp)

    # Add the camera to the scene
    scene.camera = camera
