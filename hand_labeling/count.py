import json
import os
import shutil

import cv2

path = "/Users/dawars/Desktop/processed"
path_to = "/Users/dawars/Desktop/11k_joints"

yes = 0
no = 0

accepted = []
for img in sorted(os.listdir(path)):
    # if os.path.isdir(os.path.join(path, "Hands", img)): continue
    if not img.endswith(".out"): continue
    with open(os.path.join(path, img), 'r') as f:
        char = f.read(1)
        if char == 'y':
            yes += 1
            accepted.append(img[:-4])
            shutil.copyfile(os.path.join(path, img[:-4]), os.path.join(path, path_to, img[:-4]))
        else:
            no += 1

with open(os.path.join(path, 'accepted.txt'), 'w') as f:
    for name in accepted:
        f.write(name)
        f.write('\n')

print("{} accepted, {} discarded".format(yes, no))
