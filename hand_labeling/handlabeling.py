import numpy as np
import cv2
import os
import sys

path = "/Users/dawars/Desktop/processed"

for img in sorted(os.listdir(path)):
    # if os.path.isdir(os.path.join(path, "Hands", img)): continue
    if not img.endswith(".jpg"): continue

    print(img)

    # check if already processed
    out_file = os.path.join(path, img + '.out')
    if os.path.isfile(out_file):
        print("{0} already processed".format(img))
        continue

    frame = cv2.imread(os.path.join(path, img))

    cv2.imshow('Hand labeling', frame)

    # show opencv window and wait for response
    while True:
        resp = cv2.waitKey(0)
        if resp == ord('y'):
            with open(out_file, 'w') as f:
                f.write('y')
            break
        elif resp == ord('n'):
            with open(out_file, 'w') as f:
                f.write('n')
            break
        elif resp == 27:  # esc
            sys.exit()
        else:
            print(resp)

print("Done!!!!!!")
