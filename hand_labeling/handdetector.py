import numpy as np
import cv2
import sys
import os
import json

# Load openpose.
sys.path.append('./openpose/build/python/openpose')
from openpose import *

params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
params["num_gpu_start"] = 0
params["disable_blending"] = False
params["default_model_folder"] = "/home/dawars/projects/openpose/models/"

openpose = OpenPose(params)

path = "/home/dawars/datasets/handvr/11khands/"

for img in sorted(os.listdir(os.path.join(path, "Hands"))):
    if os.path.isdir(os.path.join(path, "Hands", img)): continue

    print(img)
    frame = cv2.imread(os.path.join(path, "Hands", img))
    # frame = cv2.imread("h5.jpg")

    hands_rectangles = [[0, 0, frame.shape[1], frame.shape[0]], [0, 0, frame.shape[1], frame.shape[0]]]
    #
    # for box in hands_rectangles[0]:
    #     cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), (77, 255, 9), 3, 1)
    #

    left_hands, right_hands, frame = openpose.forward_hands(frame, hands_rectangles, True)

    joints = {'left': left_hands.tolist(), 'right': right_hands.tolist()}

    cv2.imwrite(os.path.join(path, "openpose", img), frame)

    with open(os.path.join(path, "openpose", img + '.json'), 'w') as f:
        f.write(json.dumps(joints))

print("Done!!!!!!")
