"""Code for animation the motion prediction."""

import logging
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import imageio
import h5py

IN_COLAB = 'google.colab' in sys.modules
if not IN_COLAB:
    from utils.viz import Ax3DPose
    from utils.forward_kinematics import kinematic_tree_variables
    from utils.forward_kinematics import revert_coordinate_space
    from utils.forward_kinematics import fkl
    from parsers import animation_parser
else:
    from src.utils.viz import Ax3DPose
    from src.utils.forward_kinematics import kinematic_tree_variables
    from src.utils.forward_kinematics import revert_coordinate_space
    from src.utils.forward_kinematics import fkl


def create_gif(input_dir, output_dir, filename='animation.gif'):
    """Create a gif from the frames.
    
    Parameters
    ----------
    input_dir : str
        Directory where the frames are.
    output_dir : str
        Directory where the gif will be saved.
    filename : str
        The name of the output file.
    """

    # Load all the frames
    images = []
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(input_dir, file_name)
            images.append(imageio.imread(file_path))

    # Make a pause at the end
    for _ in range(10):
        images.append(imageio.imread(file_path))

    # If folder does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save them as frames into a gif
    imageio.mimsave(os.path.join(output_dir, filename), images, duration=0.03)


def animate(args):
    """Animate the pose.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from the parser.
    """

    # If output folder does not exist, create it
    if not os.path.exists(args.imgs_dir):
        os.makedirs(args.imgs_dir)

    # Logging
    logging.basicConfig(format='%(levelname)s: %(message)s', level=20)

    # Load all the data
    parent, offset, rot_ind, expmap_ind = kinematic_tree_variables()
    with h5py.File('samples.h5', 'r') as h5f:
        # Ground truth (exponential map)
        expmap_gt = h5f['expmap/gt/walking_{}'.format(args.sample_id)][:]
        # Prediction (exponential map)
        expmap_pred = h5f['expmap/preds/walking_{}'.format(args.sample_id)][:]

    # Number of Ground truth/Predicted frames
    nframes_gt, nframes_pred = expmap_gt.shape[0], expmap_pred.shape[0]
    logging.info(f'{nframes_gt} {nframes_pred}')

    # Put them together and revert the coordinate space
    expmap_all = revert_coordinate_space(np.vstack((expmap_gt, expmap_pred)),
                                         np.eye(3), np.zeros(3))
    expmap_gt = expmap_all[:nframes_gt, :]
    expmap_pred = expmap_all[nframes_gt:, :]

    # Use forward kinematics to compute 33 3d points for each frame
    xyz_gt, xyz_pred = np.zeros((nframes_gt, 96)), np.zeros((nframes_pred, 96))

    for i in range(nframes_gt):
        xyz_gt[i, :] = fkl(expmap_gt[i, :], parent, offset, rot_ind,
                           expmap_ind)

    for i in range(nframes_pred):
        xyz_pred[i, :] = fkl(expmap_pred[i, :], parent, offset, rot_ind,
                             expmap_ind)

    # === Plot and animate ===
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ob = Ax3DPose(ax)

    # First, plot the conditioning ground truth
    for i in range(nframes_gt):
        ob.update(xyz_gt[i, :], lcolor='#ff0000', rcolor='#0000ff')
        fig.savefig(os.path.join(args.imgs_dir, f'gt_{i:02d}.png'))

    # Plot the prediction
    for i in range(nframes_pred):
        ob.update(xyz_pred[i, :], lcolor='#9b59b6', rcolor='#2ecc71')
        fig.savefig(os.path.join(args.imgs_dir, f'pred_{i:02d}.png'))


if __name__ == '__main__':
    # Load parser arguments
    args = animation_parser()

    # Animation function
    animate(args)
