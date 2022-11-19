"""Sequence-to-sequence model for human motion prediction."""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MotionPredictor(nn.Module):
    """Sequence-to-sequence model for human motion prediction"""

    def __init__(
            self,
            source_seq_len,
            target_seq_len,
            rnn_size,  # recurrent layer hidden size
            batch_size,
            learning_rate,
            learning_rate_decay_factor,
            number_of_actions,
            dropout=0.3):
        """Constructor of the class.
        
        Parameters
        ----------
        source_seq_len: int
            Length of the input sequence.
        target_seq_len: int
            Length of the target sequence.
        rnn_size: int
            Number of units in the rnn.
        batch_size: int
            The size of the batches used during training; the model
            construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g.,
            for decoding.
        learning_rate: float
            Learning rate to start with.
        learning_rate_decay_factor: 
            Decay learning rate by this much when needed.
        number_of_actions: int
            Number of classes we have.
        """

        super(MotionPredictor, self).__init__()

        self.human_dofs = 54
        self.input_size = self.human_dofs + number_of_actions

        logging.info(f'Input size is {self.input_size}')
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.dropout = dropout

        # Create the RNN that will summarize the state
        self.cell = torch.nn.GRUCell(self.input_size, self.rnn_size)
        self.fc1 = nn.Linear(self.rnn_size, self.input_size)

    def forward(self, encoder_inputs, decoder_inputs, device):
        """Forward pass of the model.

        Parameters
        ----------
        encoder_inputs : torch.Tensor
            The input to the encoder.
        decoder_inputs : torch.Tensor
            The input to the decoder.
        device : torch.device
            The device on which to do the computation.
        
        Returns
        -------
        outputs : torch.Tensor
            The transposed output of the model.
        """

        def loop_function(prev, i):
            return prev

        batch_size = encoder_inputs.shape[0]
        # To pass these data through a RNN we need to switch the first
        # two dimensions
        encoder_inputs = torch.transpose(encoder_inputs, 0, 1)
        decoder_inputs = torch.transpose(decoder_inputs, 0, 1)
        state = torch.zeros(batch_size, self.rnn_size).to(device)

        # Encoding
        for i in range(self.source_seq_len - 1):
            # Apply the RNN cell
            state = self.cell(encoder_inputs[i], state)

            # Apply dropout in training
            state = F.dropout(state, self.dropout, training=self.training)

        output_list = []

        possible_path_number = 10

        for path in range(possible_path_number):
            input_state = state + torch.randn(state.shape)
            outputs = []
            prev = None
            # Decoding, sequentially
            for i, inp in enumerate(decoder_inputs):
                # Use teacher forcing?
                if prev is not None:
                    inp = loop_function(prev, i)
                #inp = inp.detach()

                input_state = self.cell(inp, input_state)

                # Output is seen as a residual to the previous value
                output = inp + self.fc1(
                    F.dropout(input_state, self.dropout, training=self.training))
                outputs.append(output.view([1, batch_size, self.input_size]))
                prev = output
            
            outputs = torch.cat(outputs, 0)

            # Size should be batch_size x target_seq_len x input_size
            outputs = torch.transpose(outputs, 0, 1)

            output_list.append(outputs)

        return output_list

    def get_batch(self, data, actions, device):
        """Get a random batch of data from the specified bucket, prepare
        for step.
        
        Parameters
        ----------
        data:
            A list of sequences of size n-by-d to fit the model to.
        actions:
            A list of the actions we are using
        device:
            The device on which to do the computation (cpu/gpu)
        
        Returns
        -------
        encoder_inputs : torch.Tensor
            The constructed batches have the proper format to call
            step(...) later.
        decoder_inputs : torch.Tensor
            The constructed batches have the proper format to call
            step(...) later.
        target_weights : torch.Tensor
            The constructed batches have the proper format to call
            step(...) later.
        """

        # Select entries at random
        all_keys = list(data.keys())
        chosen_keys = np.random.choice(len(all_keys), self.batch_size)

        # How many frames in total do we need?
        total_frames = self.source_seq_len + self.target_seq_len
        encoder_inputs = np.zeros(
            (self.batch_size, self.source_seq_len - 1, self.input_size),
            dtype=float)
        decoder_inputs = np.zeros(
            (self.batch_size, self.target_seq_len, self.input_size),
            dtype=float)
        decoder_outputs = np.zeros(
            (self.batch_size, self.target_seq_len, self.input_size),
            dtype=float)

        # Generate the sequences
        for i in range(self.batch_size):
            the_key = all_keys[chosen_keys[i]]

            # Get the number of frames
            n, _ = data[the_key].shape

            # Sample somewhere in the middle
            idx = np.random.randint(16, n - total_frames)

            # Select the data around the sampled points
            data_sel = data[the_key][idx:idx + total_frames, :]

            # Add the data
            encoder_inputs[i, :, 0:self.input_size] = data_sel[
                0:self.source_seq_len - 1, :]
            decoder_inputs[i, :, 0:self.input_size] = data_sel[
                self.source_seq_len - 1:self.source_seq_len +
                self.target_seq_len - 1, :]
            decoder_outputs[i, :,
                            0:self.input_size] = data_sel[self.source_seq_len:,
                                                          0:self.input_size]

        encoder_inputs = torch.tensor(encoder_inputs).float().to(device)
        decoder_inputs = torch.tensor(decoder_inputs).float().to(device)
        decoder_outputs = torch.tensor(decoder_outputs).float().to(device)

        return encoder_inputs, decoder_inputs, decoder_outputs

    def find_indices_srnn(self, data, action):
        """Find the same action indices as in SRNN.
        
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325

        Parameters
        ----------
        data:
            A list of sequences.
        action:
            The action.
        
        Returns
        -------
        idx : list
            A list of indices where the action is found.
        """

        # Used a fixed dummy seed, following
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
        SEED = 1234567890
        rng = np.random.RandomState(SEED)

        subject = 5
        subaction1 = 1
        subaction2 = 2

        T1 = data[(subject, action, subaction1, 'even')].shape[0]
        T2 = data[(subject, action, subaction2, 'even')].shape[0]
        prefix, suffix = 50, 100

        # Test is performed always on subject 5
        # Select 8 random sub-sequences (by specifying their indices)
        idx = []
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))

        return idx

    def get_batch_srnn(self, data, action, device):
        """Get a random batch of data from the specified bucket,
        prepare for step.

        Parameters
        ----------
        data: dict
            Dictionary with k:v, k=((subject, action, subsequence, 'even')),
            v=nxd matrix with a sequence of poses.
        action: str
            The action to load data from, e.g. 'walking'.
        
        Returns
        -------
        encoder_inputs : torch.Tensor
            The constructed batches have the proper format to call
            step(...) later.
        decoder_inputs : torch.Tensor
            The constructed batches have the proper format to call
            step(...) later.
        target_weights : torch.Tensor
            The constructed batches have the proper format to call
            step(...) later.
        """

        actions = [
            'directions', 'discussion', 'eating', 'greeting', 'phoning',
            'posing', 'purchases', 'sitting', 'sittingdown', 'smoking',
            'takingphoto', 'waiting', 'walking', 'walkingdog',
            'walkingtogether'
        ]

        if not action in actions:
            raise ValueError(f'Unrecognized action {action}')

        frames = {}
        frames[action] = self.find_indices_srnn(data, action)

        batch_size = 8  # we always evaluate 8 sequences
        subject = 5  # we always evaluate on subject 5
        source_seq_len = self.source_seq_len
        target_seq_len = self.target_seq_len

        seeds = [(action, (i % 2) + 1, frames[action][i])
                 for i in range(batch_size)]

        encoder_inputs = np.zeros(
            (batch_size, source_seq_len - 1, self.input_size), dtype=float)
        decoder_inputs = np.zeros(
            (batch_size, target_seq_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros(
            (batch_size, target_seq_len, self.input_size), dtype=float)

        # Compute the number of frames needed
        total_frames = source_seq_len + target_seq_len

        # Reproducing SRNN's sequence subsequence selection as done in
        # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
        for i in range(batch_size):
            _, subsequence, idx = seeds[i]
            idx = idx + 50

            data_sel = data[(subject, action, subsequence, 'even')]
            data_sel = data_sel[(idx - source_seq_len):(idx +
                                                        target_seq_len), :]

            encoder_inputs[i, :, :] = data_sel[0:source_seq_len - 1, :]
            decoder_inputs[i, :, :] = data_sel[source_seq_len -
                                               1:(source_seq_len +
                                                  target_seq_len - 1), :]
            decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]

        encoder_inputs = torch.tensor(encoder_inputs).float().to(device)
        decoder_inputs = torch.tensor(decoder_inputs).float().to(device)
        decoder_outputs = torch.tensor(decoder_outputs).float().to(device)

        return encoder_inputs, decoder_inputs, decoder_outputs
