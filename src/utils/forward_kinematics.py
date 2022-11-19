"""Script with functions to compute kinematics."""

import copy

import numpy as np

IN_COLAB = 'google.colab' in sys.modules
if not IN_COLAB:
    from utils.data_utils import expmap_to_rotmat
    from utils.data_utils import rotmat_to_expmap
else:
    from src.utils.data_utils import expmap_to_rotmat
    from src.utils.data_utils import rotmat_to_expmap


def fkl(angles, parent, offset, rot_ind, expmap_ind):
    """Convert joint angles and bone lenghts into the 3d points of a person.

    Based on expmap2xyz.m, available at
    https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

    Parameters
    ----------
    angles : np.array
        99-long vector with 3d position and 3d joint angles in expmap format
    parent : np.array
        32-long vector with parent-child relationships in the kinematic tree
    offset: np.array
        96-long vector with bone lenghts
    rot_ind: np.array
        32-long list with indices into angles
    expmap_ind: np.array
        32-long list with indices into expmap angles

    Returns
    -------
    xyz: np.array
        32x3 3d points that represent a person in 3d space
    """

    assert len(angles) == 99

    # Structure that indicates parents for each joint
    njoints = 32
    xyz_struct = [dict() for x in range(njoints)]

    for i in np.arange(njoints):

        if not rot_ind[i]:  # If the list is empty
            xangle, yangle, zangle = 0, 0, 0
        else:
            xangle = angles[rot_ind[i][0] - 1]
            yangle = angles[rot_ind[i][1] - 1]
            zangle = angles[rot_ind[i][2] - 1]

        r = angles[expmap_ind[i]]

        this_rotation = expmap_to_rotmat(r)
        this_position = np.array([xangle, yangle, zangle])

        if parent[i] == -1:  # Root node
            xyz_struct[i]['rotation'] = this_rotation
            xyz_struct[i]['xyz'] = np.reshape(offset[i, :],
                                              (1, 3)) + this_position
        else:
            xyz_struct[i]['xyz'] = (offset[i, :] + this_position).dot(
                xyz_struct[parent[i]]['rotation']) + xyz_struct[
                    parent[i]]['xyz']
            xyz_struct[i]['rotation'] = this_rotation.dot(
                xyz_struct[parent[i]]['rotation'])

    xyz = [xyz_struct[i]['xyz'] for i in range(njoints)]
    xyz = np.array(xyz).squeeze()
    xyz = xyz[:, [0, 2, 1]]
    # xyz = xyz[:,[2,0,1]]

    return np.reshape(xyz, [-1])


def revert_coordinate_space(channels, R0, T0):
    """Bring a series of poses to a canonical form so they are
    facing the camera when they start.

    Adapted from
    https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

    Parameters
    ----------
    channels : np.array
        n-by-99 matrix of poses
    R0: np.array
        3x3 rotation for the first frame
    T0: np.array
        1x3 position for the first frame

    Returns
    -------
    channels_rec: np.array
        The passed poses, but the first has T0 and R0, and the
        rest of the sequence is modified accordingly.
    """

    n, d = channels.shape

    channels_rec = copy.copy(channels)
    R_prev = R0
    T_prev = T0
    rootRotInd = np.arange(3, 6)

    # Loop through the passed posses
    for i in range(n):
        R_diff = expmap_to_rotmat(channels[i, rootRotInd])
        R = R_diff.dot(R_prev)

        channels_rec[i, rootRotInd] = rotmat_to_expmap(R)
        T = T_prev + (
            (R_prev.T).dot(np.reshape(channels[i, :3], [3, 1]))).reshape(-1)
        channels_rec[i, :3] = T
        T_prev = T
        R_prev = R

    return channels_rec


def kinematic_tree_variables():
    """We define some variables that are useful to run the kinematic tree

    Returns
    -------
    parent: np.array
        32-long vector with parent-child relationships in the kinematic tree
    offset: np.array
        96-long vector with bone lenghts
    rotInd: np.array
        32-long list with indices into angles
    expmap_ind: np.array
        32-long list with indices into expmap angles
    """

    parent = np.array([
        0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13, 17, 18, 19,
        20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31
    ]) - 1

    offset = np.array([
        0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000,
        0.000000, -442.894612, 0.000000, 0.000000, -454.206447, 0.000000,
        0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437,
        132.948826, 0.000000, 0.000000, 0.000000, -442.894413, 0.000000,
        0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
        0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000,
        233.383263, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000,
        121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000,
        257.077681, 0.000000, 0.000000, 151.034226, 0.000000, 0.000000,
        278.882773, 0.000000, 0.000000, 251.733451, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000,
        100.000188, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000,
        278.892924, 0.000000, 0.000000, 251.728680, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 99.999888, 0.000000,
        137.499922, 0.000000, 0.000000, 0.000000, 0.000000
    ])
    offset = offset.reshape(-1, 3)

    rotInd = [[5, 6, 4], [8, 9, 7], [11, 12, 10], [14, 15, 13], [17, 18, 16],
              [], [20, 21, 19], [23, 24, 22], [26, 27, 25], [29, 30, 28], [],
              [32, 33, 31], [35, 36, 34], [38, 39, 37], [41, 42, 40], [],
              [44, 45, 43], [47, 48, 46], [50, 51, 49], [53, 54, 52],
              [56, 57, 55], [], [59, 60, 58], [], [62, 63, 61], [65, 66, 64],
              [68, 69, 67], [71, 72, 70], [74, 75, 73], [], [77, 78, 76], []]

    expmap_ind = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, rotInd, expmap_ind
