"""Code for training an RNN for motion prediction."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from parsers import training_parser
from utils.data_utils import read_all_data
from utils.data_utils import define_actions
from models.motionpredictor import MotionPredictor


def train(args):
    """Train a seq2seq model on human motion.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from the parser.
    """

    # Set of actions
    actions = define_actions(args.action)
    number_of_actions = len(actions)

    train_set, test_set, _, _, _, _ = read_all_data(actions,
                                                    args.seq_length_in,
                                                    args.seq_length_out,
                                                    args.data_dir)

    # Create model for training only
    model = MotionPredictor(
        args.seq_length_in,
        args.seq_length_out,
        args.size,  # hidden layer size
        args.batch_size,
        args.learning_rate,
        args.learning_rate_decay_factor,
        len(actions))
    model = model.to(device)

    # This is the training loop
    loss, val_loss = 0.0, 0.0
    current_step = 0
    all_losses = []
    all_val_losses = []

    # The optimizer
    #optimiser = optim.SGD(model.parameters(), lr=args.learning_rate)
    optimiser = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           betas=(0.9, 0.999))

    for _ in range(args.iterations):
        optimiser.zero_grad()
        # Set a flag to compute gradients
        model.train()

        # === Training step ===
        # Get batch from the training set
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(
            train_set, actions, device)

        # Forward pass
        preds = model(encoder_inputs, decoder_inputs, device)

        # Loss: Mean Squared Errors
        step_loss = (preds - decoder_outputs)**2
        step_loss = step_loss.mean()

        # Backpropagation
        step_loss.backward()

        # Gradient descent step
        optimiser.step()

        step_loss = step_loss.cpu().data.numpy()

        if current_step % 10 == 0:
            logging.info(f'step {current_step:04}; step_loss: {step_loss:.4f}')
        loss += step_loss / args.test_every
        current_step += 1

        # === step decay ===
        if current_step % args.learning_rate_step == 0:
            args.learning_rate = args.learning_rate * args.learning_rate_decay_factor
            optimiser = optim.Adam(model.parameters(),
                                   lr=args.learning_rate,
                                   betas=(0.9, 0.999))
            print('Decay learning rate. New value at {args.learning_rate}')

        # Once in a while, save checkpoint, print statistics.
        if current_step % args.test_every == 0:
            model.eval()
            # === Validation ===
            encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(
                test_set, actions, device)
            preds = model(encoder_inputs, decoder_inputs, device)

            step_loss = (preds - decoder_outputs)**2
            val_loss = step_loss.mean()

            print('\n=================================\n'
                  f'Global step:         {current_step}\n'
                  f'Learning rate:       {args.learning_rate:.4}\n'
                  f'Train loss avg:      {loss:.4}\n'
                  '-------------------------------\n'
                  f'Val loss:            {val_loss:.4}\n'
                  '=================================\n')
            all_val_losses.append(
                [current_step, val_loss.cpu().detach().numpy()])
            all_losses.append([current_step, loss])
            torch.save(model, train_dir + '/model_' + str(current_step))

            # Reset loss
            loss = 0

    vlosses = np.array(all_val_losses)
    tlosses = np.array(all_losses)

    # Plot losses
    plt.plot(vlosses[:, 0], vlosses[:, 1], 'b')
    plt.plot(tlosses[:, 0], tlosses[:, 1], 'r')
    plt.legend(['Validation loss', 'Training loss'])
    plt.show()


if __name__ == '__main__':
    # Load parser
    args = training_parser()

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

    logging.info('Train dir: ' + train_dir)
    os.makedirs(train_dir, exist_ok=True)

    # Training function
    train(args)
