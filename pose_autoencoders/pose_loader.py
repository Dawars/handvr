"""
Loading hand pose dataset
"""
import numpy as np

from utils.mano_utils import mano_data


def get_poses():
    """
    Return a Tensor containing the normalized poses (add 'hands_mean' to get real values)
    (1554, 45) - 3*15 = 45 joint anles for 1554 people
    """

    hands_components = mano_data['hands_components']
    hands_coeffs = mano_data['hands_coeffs']

    # 3*15 = 45 joint angles
    hands_poses = np.matmul(hands_coeffs, hands_components)

    return hands_poses


if __name__ == '__main__':
    poses = get_poses()
    print(poses.shape)
    print(poses.max())
    print(poses.min())
