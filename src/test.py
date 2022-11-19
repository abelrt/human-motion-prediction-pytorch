"""Code for training an RNN for motion prediction."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py

IN_COLAB = 'google.colab' in sys.modules
if not IN_COLAB:
    from parsers import testing_parser
    from utils.data_utils import read_all_data
    from utils.data_utils import define_actions
    from utils.data_utils import rotmat_to_expmap
    from utils.data_utils import expmap_to_rotmat
    from utils.data_utils import unnormalize_data
    from utils.data_utils import revert_output_format
    from utils.evaluation import evaluate_batch
else:
    from src.utils.data_utils import read_all_data
    from src.utils.data_utils import define_actions
    from src.utils.data_utils import rotmat_to_expmap
    from src.utils.data_utils import expmap_to_rotmat
    from src.utils.data_utils import unnormalize_data
    from src.utils.data_utils import revert_output_format
    from src.utils.evaluation import evaluate_batch


def get_srnn_gts(actions,
                 model,
                 test_set,
                 data_mean,
                 data_std,
                 dim_to_ignore,
                 to_euler=True):
    """Get the ground truths for srnn's sequences, and convert to Euler
    angles (the error is always computed in Euler angles).

    Parameters
    ----------
    actions : list
        A list of actions to get ground truths for.
    model: torch.nn.Module
        Training model we are using (we only use the "get_batch" method).
    test_set: dict
        Dictionary with normalized training data.
    data_mean: np.array
        d-long vector with the mean of the training data.
    data_std: np.array
        d-long vector with the standard deviation of the training data.
    dim_to_ignore: np.array
        Dimensions that we are not using to train/predict.
    to_euler: bool
        Whether to convert the angles to Euler format or keep thm in
        exponential map.

    Returns
    -------
    srnn_gts_euler:
        A dictionary where the keys are actions, and the values are the
        ground_truth, denormalized expected outputs of srnns's seeds.
    """

    srnn_gts_euler = {}

    for action in actions:
        srnn_gt_euler = []
        # get_batch or get_batch_srnn
        _, _, srnn_expmap = model.get_batch_srnn(test_set, action, device)
        srnn_expmap = srnn_expmap.cpu()
        # expmap -> rotmat -> euler
        for i in np.arange(srnn_expmap.shape[0]):
            denormed = unnormalize_data(srnn_expmap[i, :, :], data_mean,
                                        data_std, dim_to_ignore, actions)
            if to_euler:
                for j in np.arange(denormed.shape[0]):
                    for k in np.arange(3, 97, 3):
                        denormed[j, k:k + 3] = rotmat_to_expmap(
                            expmap_to_rotmat(denormed[j, k:k + 3]))
            srnn_gt_euler.append(denormed)

        # Put back in the dictionary
        srnn_gts_euler[action] = srnn_gt_euler
    return srnn_gts_euler


def test(args):
    """Sample predictions for srnn's seeds.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from the parser.
    """

    # Set logger
    if args.log_file == '':
        logging.basicConfig(format='%(levelname)s: %(message)s',
                            level=args.log_level)
    else:
        logging.basicConfig(filename=args.log_file,
                            format='%(levelname)s: %(message)s',
                            level=args.log_level)

    # Set directory
    train_dir = os.path.normpath(
        os.path.join(args.train_dir, args.action, f'out_{args.seq_length_out}',
                     f'iterations_{args.iterations}', f'size_{args.size}',
                     f'lr_{args.learning_rate}'))

    # Detect device
    if torch.cuda.is_available():
        logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        logging.info('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('Test dir: ' + train_dir)
    os.makedirs(train_dir, exist_ok=True)

    # Set of actions
    actions = define_actions(args.action)
    nsamples = 8

    # Create the model
    logging.info(f'Creating a model with {args.size} units.')
    sampling = True
    logging.info('Loading model')
    model = torch.load(train_dir + '/model_' + str(args.load_model))
    model.source_seq_len = 50
    model.target_seq_len = 100
    model = model.to(device)
    logging.info('Model created')

    # Load all the data
    _, test_set, data_mean, data_std, dim_to_ignore, _ = read_all_data(
        actions, 50, args.seq_length_out, args.data_dir)

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_expmap = get_srnn_gts(actions,
                                   model,
                                   test_set,
                                   data_mean,
                                   data_std,
                                   dim_to_ignore,
                                   to_euler=False)
    srnn_gts_euler = get_srnn_gts(actions, model, test_set, data_mean,
                                  data_std, dim_to_ignore)

    # Clean and create a new h5 file of samples
    SAMPLES_FNAME = 'samples.h5'
    try:
        os.remove(SAMPLES_FNAME)
    except OSError:
        pass

    # Predict and save for each action
    for action in actions:

        # Make prediction with srnn' seeds
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn(
            test_set, action, device)
        # Forward pass
        srnn_poses = model(encoder_inputs, decoder_inputs, device)
        srnn_loss = (srnn_poses - decoder_outputs)**2
        srnn_loss.cpu().data.numpy()
        srnn_loss = srnn_loss.mean()
        srnn_poses = srnn_poses.cpu().data.numpy()
        srnn_poses = srnn_poses.transpose([1, 0, 2])
        srnn_loss = srnn_loss.cpu().data.numpy()

        # Restores the data in the same format as the original: dimension 99.
        # Returns a tensor of size (batch_size, seq_length, dim) output.
        srnn_pred_expmap = revert_output_format(srnn_poses, data_mean,
                                                data_std, dim_to_ignore,
                                                actions)

        # Save the samples
        with h5py.File(SAMPLES_FNAME, 'a') as hf:
            for i in np.arange(nsamples):
                # Save conditioning ground truth
                node_name = f'expmap/gt/{action}_{i}'
                hf.create_dataset(node_name, data=srnn_gts_expmap[action][i])
                # Save prediction
                node_name = f'expmap/preds/{action}_{i}'
                hf.create_dataset(node_name, data=srnn_pred_expmap[i])

        # Compute and save the errors here
        mean_errors_batch = evaluate_batch(srnn_pred_expmap,
                                           srnn_gts_euler[action])
        logging.info(
            'Mean error for test data along the horizon on action {}: {}'.
            format(action, mean_errors_batch))
        logging.info(
            'Mean error for test data at horizon {} on action {}: {}'.format(
                args.horizon_test_step, action,
                mean_errors_batch[args.horizon_test_step]))
        with h5py.File(SAMPLES_FNAME, 'a') as hf:
            node_name = f'mean_{action}_error'
            hf.create_dataset(node_name, data=mean_errors_batch)
    return


if __name__ == "__main__":
    # Load parser
    args = testing_parser()

    # Testing function
    test(args)
