"""Functions that help with data processing for human3.6m"""

from six.moves import xrange  # pylint: disable=redefined-builtin
import logging
import copy

import numpy as np


def rotmat_to_euler(R):
    """Converts a rotation matrix to Euler angles.
    
    Matlab port to python for evaluation purposes.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

    Parameters
    ----------
    R: np.array
        A 3x3 rotation matrix.
    
    Returns
    -------
    eul: np.array
        A 3x1 Euler angle representation of R.
    """

    if R[0, 2] == 1 or R[0, 2] == -1:
        # special case
        E3 = 0  # set arbitrarily
        dlta = np.arctan2(R[0, 1], R[0, 2])

        if R[0, 2] == -1:
            E2 = np.pi / 2
            E1 = E3 + dlta
        else:
            E2 = -np.pi / 2
            E1 = -E3 + dlta
    else:
        E2 = -np.arcsin(R[0, 2])
        E1 = np.arctan2(R[1, 2] / np.cos(E2), R[2, 2] / np.cos(E2))
        E3 = np.arctan2(R[0, 1] / np.cos(E2), R[0, 0] / np.cos(E2))

    eul = np.array([E1, E2, E3])

    return eul


def quat_to_expmap(q):
    """Converts a quaternion to an exponential map.
    
    Matlab port to python for evaluation purposes.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

    Parameters
    ----------
    q: np.array
        1x4 quaternion.
    
    Returns
    -------
    r: np.array
        1x3 exponential map.
    
    Raises
    ------
    ValueError 
        If the l2 norm of the quaternion is not close to 1.
    """

    if (np.abs(np.linalg.norm(q) - 1) > 1e-3):
        raise (ValueError, 'quat_to_expmap: input quaternion is not norm 1')

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]

    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0

    r = r0 * theta
    return r


def rotmat_to_quat(R):
    """Converts a rotation matrix to a quaternion.
    
    Matlab port to python for evaluation purposes.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

    Parameters
    ----------
    R: np.array
        3x3 rotation matrix.
    
    Returns
    -------
    q: np.array
        1x4 quaternion
    """

    rotdiff = R - R.T

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]

    sintheta = np.linalg.norm(r) / 2
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)

    costheta = (np.trace(R) - 1) / 2
    theta = np.arctan2(sintheta, costheta)

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)

    return q


def rotmat_to_expmap(R):
    """Converts a rotation matrix to an exponential map.

    Parameters
    ----------
    R: np.array
        3x3 rotation matrix.
    
    Returns
    -------
    r: np.array
        1x3 exponential map.
    """

    r = quat_to_expmap(rotmat_to_quat(R))

    return r


def expmap_to_rotmat(r):
    """Converts an exponential map angle to a rotation matrix.
    
    Matlab port to python for evaluation purposes.
    I believe this is also called Rodrigues' formula.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

    Parameters
    ----------
    r: np.array
        1x3 exponential map.
    
    Returns
    -------
    R: np.array
        3x3 rotation matrix.
    """

    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x \
        + (1 - np.cos(theta)) * (r0x).dot(r0x)

    return R


def unnormalize_data(normalized_data, data_mean, data_std,
                     dimensions_to_ignore, actions):
    """Reads a csv file and returns a float32 matrix.

    Borrowed from SRNN code.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

    Parameters
    ----------
    normalized_data: np.array
        nxd matrix with normalized data.
    data_mean: np.array
        Vector of mean used to normalize the data.
    data_std: np.array
        Vector of standard deviation used to normalize the data.
    dimensions_to_ignore: np.array
        Vector with dimensions not used by the model.
    actions: np.array
        List of strings with the encoded actions.
    
    Returns
    -------
    orig_data: np.array
        Data originally used.
    """

    T = normalized_data.shape[0]
    D = data_mean.shape[0]

    orig_data = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    orig_data[:, dimensions_to_use] = normalized_data[:, :-len(actions)]

    # potentially ineficient, but only done once per experiment
    std_mat = data_std.reshape((1, D))
    std_mat = np.repeat(std_mat, T, axis=0)
    mean_mat = data_mean.reshape((1, D))
    mean_mat = np.repeat(mean_mat, T, axis=0)
    orig_data = np.multiply(orig_data, std_mat) + mean_mat

    return orig_data


def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions):
    """Converts the output of the neural network to a format that is
    easier to manipulate for, e.g. conversion to other format or
    visualization.

    Parameters
    ----------
    poses: np.array
        A list with (seq_length) entries, each with a (batch_size, dim)
        output.
    
    Returns
    -------
    poses_out: np.array
        A tensor of size (batch_size, seq_length, dim) output. Each
        batch is an n-by-d sequence of poses.
    """

    seq_len = len(poses)
    if seq_len == 0:
        return []

    batch_size, dim = poses[0].shape

    poses_out = np.concatenate(poses)
    poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
    poses_out = np.transpose(poses_out, [1, 0, 2])
    poses_out_list = []
    for i in xrange(poses_out.shape[0]):
        poses_out_list.append(
            unnormalize_data(poses_out[i, :, :], data_mean, data_std,
                             dim_to_ignore, actions))
    return poses_out_list


def read_csv_as_float(filename):
    """Reads a csv and returns a float matrix.

    Borrowed from SRNN code.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Parameters
    ----------
    filename: str
        Path to the csv file.
    
    Returns
    -------
    return_array: np.array
        The read data in a float32 matrix.
    """

    return_array = []
    lines = open(filename).readlines()

    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            return_array.append(np.array([np.float32(x) for x in line]))

    return np.array(return_array)


def load_data(path_to_dataset, subjects, actions):
    """This is how the SRNN code reads the provided .txt files.

    Borrowed from SRNN code.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270

    Parameters
    ----------
    path_to_dataset: str    
        Directory where the data resides.
    subjects: list
        The subjects to load.
    actions: list
        A list of strings with the actions to load.
    
    Returns
    -------
    train_data: 
        A dictionary with k:v; k=(subject, action, subaction, 'even'),
        v=(nxd) un-normalized data.
    complete_data: np.array
        nxd matrix with all the data. Used to normlization stats.
    """

    n_actions = len(actions)

    train_data = {}
    complete_data = []
    for subj in subjects:
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]

            for subact in [1, 2]:  # subactions
                logging.info(
                    f'Reading subject {subj}, action {action}, subaction {subact}'
                )
                filename = '{0}/S{1}/{2}_{3}.txt'.format(
                    path_to_dataset, subj, action, subact)
                action_sequence = read_csv_as_float(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, 2)
                # Add a one-hot encoding at the end of the representation
                the_sequence = np.zeros((len(even_list), d + n_actions),
                                        dtype=float)
                the_sequence[:, 0:d] = action_sequence[even_list, :]
                the_sequence[:, d + action_idx] = 1
                train_data[(subj, action, subact, 'even')] = the_sequence

                if len(complete_data) == 0:
                    complete_data = copy.deepcopy(action_sequence)
                else:
                    complete_data = np.append(complete_data,
                                              action_sequence,
                                              axis=0)
    return train_data, complete_data


def normalize_data(data, data_mean, data_std, dim_to_use, actions):
    """Normalize input data by removing unused dimensions, subtracting
    the mean and dividing by the standard deviation.

    Parameters
    ----------
    data: np.array
        nx99 matrix with data to normalize.
    data_mean: np.array
        Vector of mean used to normalize the data.
    data_std: np.array
        Vector of standard deviation used to normalize the data.
    dim_to_use: np.array
        Vector with dimensions used by the model.
    actions: list
        A list of strings with the encoded actions.
    
    Returns
    -------
    data_out: np.array
        The passed data matrix, but normalized.
    """

    data_out = {}
    n_actions = len(actions)

    # TODO hard-coding 99 dimensions for un-normalized human poses
    for key in data.keys():
        data_out[key] = np.divide((data[key][:, 0:99] - data_mean), data_std)
        data_out[key] = data_out[key][:, dim_to_use]
        data_out[key] = np.hstack((data_out[key], data[key][:, -n_actions:]))

    return data_out


def normalization_stats(complete_data):
    """Computes mean, stdev and dimensions to ignore.

    Also borrowed for SRNN code.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

    Parameters
    ----------
    complete_data: np.array
        nx99 matrix with data to normalize.
    
    Returns
    -------
    data_mean: np.array
        A vector of mean used to normalize the data.
    data_std: np.array
        A vector of standard deviation used to normalize the data.
    dimensions_to_ignore: np.array
        A vector with dimensions not used by the model.
    dimensions_to_use: np.array
        A vector with dimensions used by the model.
    """

    data_mean = np.mean(complete_data, axis=0)
    data_std = np.std(complete_data, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []
    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))
    data_std[dimensions_to_ignore] = 1.0

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def define_actions(action):
    """
    Define the list of actions we are using.

    Args
        action: String with the passed action. Could be "all"
    Returns
        actions: List of strings of actions
    Raises
        ValueError if the action is not included in H3.6M
    """

    actions = [
        'walking', 'eating', 'smoking', 'discussion', 'directions', 'greeting',
        'phoning', 'posing', 'purchases', 'sitting', 'sittingdown',
        'takingphoto', 'waiting', 'walkingdog', 'walkingtogether'
    ]

    if action in actions:
        return [action]
    if action == 'all':
        return actions
    if action == 'all_srnn':
        return ['walking', 'eating', 'smoking', 'discussion']
    raise (ValueError, f'Unrecognized action: {action}')


def read_all_data(actions, seq_length_in, seq_length_out, data_dir):
    """Loads data for training/testing and normalizes it.

    Parameters
    ----------
    actions: list
        A list of strings (actions) to load.
    seq_length_in: int
        The number of frames to use in the burn-in sequence.
    seq_length_out: int
        The number of frames to use in the output sequence.
    data_dir: str
        The directory to load the data from.
    
    Returns
    -------
    train_set: dict
        A dictionary with normalized training data.
    test_set: dict
        A dictionary with test data.
    data_mean: np.array
        d-long vector with the mean of the training data.
    data_std: np.array
        d-long vector with the standard dev of the training data.
    dim_to_ignore: np.array
        The dimensions that are not used becaused stdev is too small.
    dim_to_use: np.array
        The dimensions that we are actually using in the model.
    """

    # === Read training data ===
    logging.info(
        "Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
            seq_length_in, seq_length_out))
    train_subject_ids = [1, 6, 7, 9, 11]
    test_subject_ids = [5]
    train_set, complete_train = load_data(data_dir, train_subject_ids, actions)
    test_set, complete_test = load_data(data_dir, test_subject_ids, actions)

    # Compute normalization stats
    data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats(
        complete_train)

    # Normalize -- subtract mean, divide by stdev
    train_set = normalize_data(train_set, data_mean, data_std, dim_to_use,
                               actions)
    test_set = normalize_data(test_set, data_mean, data_std, dim_to_use,
                              actions)

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use
