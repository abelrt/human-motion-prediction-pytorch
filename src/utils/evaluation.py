"""Script to evaluate batches of predictions."""

import numpy as np

from utils.data_utils import rotmat_to_euler
from utils.data_utils import expmap_to_rotmat


def evaluate(eulerchannels_pred, eulerchannels_gt):
    """Evaluate a single prediction.

    Parameters
    ----------
    eulerchannels_pred : np.array
        Predicted euler channels of shape (seq_len, 99)
    eulerchannels_gt : np.array
        Ground truth euler channels of shape (seq_len, 99)

    Returns
    -------
    euc_error : np.float
        Euclidean error
    """

    for j in np.arange(eulerchannels_pred.shape[0]):
        for k in np.arange(3, 97, 3):
            eulerchannels_pred[j, k:k + 3] = rotmat_to_euler(
                expmap_to_rotmat(eulerchannels_pred[j, k:k + 3]))
    eulerchannels_pred[:, 0:6] = 0

    # Pick only the dimensions with sufficient standard deviation.
    # Others are ignored.
    idx_to_use = np.where(np.std(eulerchannels_pred, 0) > 1e-4)[0]

    # Euclidean distance between Euler angles for sample i
    euc_error = np.power(
        eulerchannels_gt[:, idx_to_use] - eulerchannels_pred[:, idx_to_use], 2)
    euc_error = np.sum(euc_error, 1)
    euc_error = np.sqrt(euc_error)

    return euc_error


def evaluate_batch(euler_pred, eulerchannels_gt):
    """Evaluate a whole batch (all errors at each timestep).

    Parameters
    ----------
    euler_pred : np.array
        Predicted euler channels of shape (batch_size, seq_len, 99)
    eulerchannels_gt : np.array
        Ground truth euler channels of shape (batch_size, seq_len, 99)

    Returns
    -------
    mean_error : np.float
        Mean error.
    """

    nsamples = len(euler_pred)
    mean_errors = np.zeros((nsamples, euler_pred[0].shape[0]))

    for i in np.arange(nsamples):
        mean_errors[i, :] = evaluate(euler_pred[i], eulerchannels_gt[i])

    mean_error = np.mean(mean_errors, 0)

    return mean_error
