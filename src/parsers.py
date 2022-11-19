"""Argument parsers for the command line interface."""

import os
import argparse
from argparse import Namespace


def training_parser():
    """Argument parser for training script.

    Returns
    -------
    args : argparse.Namespace
        Arguments from the parser.
    """

    parser = argparse.ArgumentParser(
        description='Train RNN for human pose estimation')

    # Learning parameters
    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        help='Learning rate',
                        default=0.00001,
                        type=float)

    parser.add_argument('--learning_rate_decay_factor',
                        dest='learning_rate_decay_factor',
                        help='Learning rate is multiplied by this'
                        'much. 1 means no decay.',
                        default=0.95,
                        type=float)

    parser.add_argument('--learning_rate_step',
                        dest='learning_rate_step',
                        help='Every this many steps, do decay.',
                        default=10000,
                        type=int)

    parser.add_argument('--batch_size',
                        dest='batch_size',
                        help='Batch size to use during training.',
                        default=128,
                        type=int)

    parser.add_argument('--iterations',
                        dest='iterations',
                        help='Iterations to train for.',
                        default=1e5,
                        type=int)

    parser.add_argument('--test_every',
                        dest='test_every',
                        help='',
                        default=100,
                        type=int)

    parser.add_argument('--size',
                        dest='size',
                        help='Size of each model layer.',
                        default=512,
                        type=int)

    parser.add_argument('--seq_length_in',
                        dest='seq_length_in',
                        help='Number of frames to feed into'
                        'the encoder. 25 fps',
                        default=50,
                        type=int)

    parser.add_argument('--seq_length_out',
                        dest='seq_length_out',
                        help='Number of frames that the decoder'
                        'has to predict. 25fps',
                        default=10,
                        type=int)

    # Directory parameters
    parser.add_argument('--data_dir',
                        dest='data_dir',
                        help='Data directory',
                        default=os.path.normpath("./data/h3.6m/dataset"),
                        type=str)

    parser.add_argument('--train_dir',
                        dest='train_dir',
                        help='Training directory',
                        default=os.path.normpath("./experiments/"),
                        type=str)

    parser.add_argument('--action',
                        dest='action',
                        help='The action to train on. all means all the'
                        'actions, all_periodic means walking, eating'
                        'and smoking',
                        default='all',
                        type=str)

    parser.add_argument('--log-level',
                        dest='log_level',
                        type=int,
                        default=20,
                        help='Log level (default: 20)')

    parser.add_argument('--log-file',
                        dest='log_file',
                        default='',
                        help='Log file (default: standard output)')

    args = parser.parse_args()
    return args


def training_parser_from_dict(dict_args):
    """Build training parser from a dictionary.

    Parameters
    ----------
    dict_args : dict
        Dictionary with the arguments.

    Returns
    -------
    args : argparse.Namespace
        Arguments from the parser.
    """

    default_params = {
        'learning_rate': 0.00001,
        'learning_rate_decay_factor': 0.95,
        'learning_rate_step': 10000,
        'batch_size': 128,
        'iterations': int(1e5),
        'test_every': 100,
        'size': 512,
        'seq_length_in': 50,
        'seq_length_out': 10,
        'data_dir': os.path.normpath('./data/h3.6m/dataset'),
        'train_dir': os.path.normpath('./experiments/'),
        'action': 'all',
        'log_level': 20,
        'log_file': '',
    }

    default_params.update(dict_args)
    args = Namespace(**default_params)

    return args


def testing_parser():
    """Argument parser for testing script.

    Returns
    -------
    args : argparse.Namespace
        Arguments from the parser.
    """

    # Learning parameters
    parser = argparse.ArgumentParser(
        description='Test RNN for human pose estimation')

    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        help='Learning rate',
                        default=0.00001,
                        type=float)

    parser.add_argument('--batch_size',
                        dest='batch_size',
                        help='Batch size to use during training.',
                        default=128,
                        type=int)

    parser.add_argument('--iterations',
                        dest='iterations',
                        help='Iterations to train for.',
                        default=1e5,
                        type=int)

    parser.add_argument('--size',
                        dest='size',
                        help='Size of each model layer.',
                        default=512,
                        type=int)

    parser.add_argument('--seq_length_out',
                        dest='seq_length_out',
                        help='Number of frames that the decoder'
                        'has to predict. 25fps',
                        default=10,
                        type=int)

    parser.add_argument('--horizon-test-step',
                        dest='horizon_test_step',
                        help='Time step at which we evaluate the error',
                        default=25,
                        type=int)

    # Directory parameters
    parser.add_argument('--data_dir',
                        dest='data_dir',
                        help='Data directory',
                        default=os.path.normpath("./data/h3.6m/dataset"),
                        type=str)

    parser.add_argument('--train_dir',
                        dest='train_dir',
                        help='Training directory',
                        default=os.path.normpath("./experiments/"),
                        type=str)

    parser.add_argument('--action',
                        dest='action',
                        help='The action to train on. all means all the'
                        'actions, all_periodic means walking,'
                        'eating and smoking',
                        default='all',
                        type=str)

    parser.add_argument('--load-model',
                        dest='load_model',
                        help='Try to load a previous checkpoint.',
                        default=0,
                        type=int)

    parser.add_argument('--log-level',
                        dest='log_level',
                        type=int,
                        default=20,
                        help='Log level (default: 20)')

    parser.add_argument('--log-file',
                        dest='log_file',
                        default='',
                        help='Log file (default: standard output)')

    args = parser.parse_args()
    return args


def testing_parser_from_dict(dict_args):
    """Build testing parser from a dictionary.

    Parameters
    ----------
    dict_args : dict
        Dictionary with the arguments.

    Returns
    -------
    args : argparse.Namespace
        Arguments from the parser.
    """

    default_params = {
        'learning_rate': 0.00001,
        'batch_size': 128,
        'iterations': int(1e5),
        'size': 512,
        'seq_length_out': 10,
        'horizon_test_step': 25,
        'data_dir': os.path.normpath('./data/h3.6m/dataset'),
        'train_dir': os.path.normpath('./experiments/'),
        'action': 'all',
        'load_model': 0,
        'log_level': 20,
        'log_file': '',
    }

    default_params.update(dict_args)
    args = Namespace(**default_params)

    return args


def animation_parser():
    """Argument parser for animation script.

    Returns
    -------
    args : argparse.Namespace
        Arguments from the parser.
    """

    parser = argparse.ArgumentParser(
        description='Test RNN for human pose estimation')
    parser.add_argument('--id',
                        dest='sample_id',
                        help='Sample id.',
                        default=0,
                        type=int)

    args = parser.parse_args()
    return args


def animation_parser_from_dict(dict_args):
    """Build animation parser from a dictionary.

    Parameters
    ----------
    dict_args : dict
        Dictionary with the arguments.

    Returns
    -------
    args : argparse.Namespace
        Arguments from the parser.
    """

    default_params = {'sample_id': 0}

    default_params.update(dict_args)
    args = Namespace(**default_params)

    return args
