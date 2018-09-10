"""
Builds a skeleton for the selected model from the pickle file, binds it, skins it

The mean model (e.g. MANO) should be selected in Maya

This script should be run on the Maya interpreter
"""
import os
import pickle

import maya.cmds as cmds
import json

# constants
N = 778  # number of vertices
K = 15  # number of bones (+6 for wrist and finger ends)

parentOf = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]

path = './'
if os.getcwd() == '/':
    path = '/Users/dawars/projects/handvr/'


def build_rig():
    print("Please name MANO mesh as 'Mano' and select it!")

    with open(os.path.join(path, "mpi/data/mano/SMPLH_female_py3.pkl")) as f:
        model = pickle.load(f)

    joint_names = [
        "R_Wrist",
        "R_index_a", "R_index_b", "R_index_c",
        "R_middle_a", "R_middle_b", "R_middle_c",
        "R_pinky_a", "R_pinky_b", "R_pinky_c",
        "R_ring_a", "R_ring_b", "R_ring_c",
        "R_thumb_a", "R_thumb_b", "R_thumb_c"
    ]

    joints = []

    joint_positions = model['joints']

    # wrist/root
    # j_wrist = cmds.joint(position=joint_positions[0], dof="xyz", name="R_Wrist")

    for j_pos, name in zip(joint_positions, joint_names):
        joints.append(cmds.joint(position=j_pos, dof="xyz", name=name))
        cmds.select(cl=True)

    print("Joints created")

    j_wrist = joints[0]

    # finger ending
    finger_end_names = ["R_index_end", "R_middle_end", "R_pinky_end", "R_ring_end", "R_thumb_end"]
    finger_ends_pos = [[-7.274431229, 0.5078120232, 2.847385883],
                       [-7.899281979, 0.6146649718, -1.204085946],
                       [-2.368740082, -0.552932024, -6.978840828],
                       [-6.249189854, 0.2426860034, -4.066926956],
                       [0.3715699911, -1.635903001, 9.410496712], ]
    finger_end_joints = []
    for j_pos, name in zip(finger_ends_pos, finger_end_names):
        finger_end_joints.append(cmds.joint(position=j_pos, dof="xyz", name=name))
        cmds.select(cl=True)

    ## connecting joints
    for i in range(1, len(parentOf)):
        cmds.connectJoint([joints[i], joints[parentOf[i]]], parentMode=True)

    cmds.connectJoint([finger_end_joints[0], joints[3]], parentMode=True)
    cmds.connectJoint([finger_end_joints[1], joints[6]], parentMode=True)
    cmds.connectJoint([finger_end_joints[2], joints[9]], parentMode=True)
    cmds.connectJoint([finger_end_joints[3], joints[12]], parentMode=True)
    cmds.connectJoint([finger_end_joints[4], joints[15]], parentMode=True)

    print("Joints connected")

    ## reorient joints
    cmds.select([j_wrist], r=True)
    cmds.joint(edit=True, orientJoint="xzy", secondaryAxisOrient="zup", children=True, zeroScaleOrient=True)
    # joint -e  -oj xyz -secondaryAxisOrient zup -ch -zso;

    print("Joints reoriented")

    joint_orientations = []
    for joint in joints:
        cmds.select([joint], r=True)
        orient = cmds.joint(query=True, orientation=True)
        joint_orientations.append(orient)
        # print(orient)

    with open(os.path.join(path, 'mano_params/joint_orient.json'), 'w') as f:
        f.write(json.dumps(joint_orientations))

    # bind skin to skeleton
    cmds.select(joints, replace=True)
    cmds.select("Mano", toggle=True)
    skinCluster = cmds.skinCluster()[0]

    ## set weights
    with open(os.path.join(path, 'mano_params/weights.json'), 'r') as f:
        weights = json.load(f)

        for i, weight in enumerate(weights):
            # http://forums.cgsociety.org/archive/index.php?t-1198569.html
            weight_ = [(joint_names[i], w) for i, w in enumerate(weight)]
            cmds.skinPercent(skinCluster, "ManoShape.vtx[{0}]".format(i),  # vertex
                             transformValue=weight_)

    print("Blend weights set")

    # weights for end joints
    for i in range(N):  # for every vertex
        cmds.skinPercent(skinCluster, "ManoShape.vtx[{0}]".format(i),  # how much a joint influences it
                         transformValue=[(j, 0) for j in finger_end_joints])


if __name__ == '__main__':
    build_rig()
