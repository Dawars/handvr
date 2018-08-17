import sys

import tensorflow as tf
from tensorflow import keras
import torch

print("python: {}".format(sys.version))

print("pytorch: {}".format(torch.__version__))
print("tensorflow: {}".format(tf.__version__))
print("keras: {}".format(keras.__version__))
