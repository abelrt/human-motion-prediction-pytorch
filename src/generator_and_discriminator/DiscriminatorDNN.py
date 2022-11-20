"""Sequence-to-sequence model for human motion prediction."""
import random
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class DiscriminatorRNN(nn.Module):
    """Sequence-to-sequence model for human motion prediction"""
    def __init__(self,source_seq_len,target_seq_len,
        rnn_size, # recurrent layer hidden size
        batch_size,learning_rate,learning_rate_decay_factor,
        number_of_actions,dropout=0.3):

        """Args:
        source_seq_len: length of the input sequence.
        target_seq_len: length of the target sequence.
        rnn_size: number of units in the rnn.
        batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
        learning_rate: learning rate to start with.
        learning_rate_decay_factor: decay learning rate by this much when needed.
        number_of_actions: number of classes we have.
        """
        super(DiscriminatorRNN, self).__init__()

        self.human_dofs     = 54
        self.input_size     = self.human_dofs + number_of_actions

        logging.info("Input size is {}".format(self.input_size))
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.rnn_size       = rnn_size
        self.batch_size     = batch_size
        self.dropout        = dropout

        # === Create the RNN that will summarizes the state ===
        self.cell           = torch.nn.GRUCell(self.input_size, self.rnn_size)
        self.fc1            = nn.Linear(self.rnn_size, self.input_size)
        self.fc2            = nn.Linear(self.input_size*(self.target_seq_len+self.source_seq_len-1),1)
        self.fc3            = nn.Sigmoid()

    # Forward pass
    def forward(self, encoder_inputs, decoder_output, device):
        def loop_function(prev, i):
            return prev
        all_frame=torch.cat((encoder_inputs,decoder_output),1)
        batch_size     = encoder_inputs.shape[0]
        # To pass these data through a RNN we need to switch the first two dimensions
        encoder_inputs = torch.transpose(encoder_inputs, 0, 1)
        #decoder_inputs = torch.transpose(decoder_inputs, 0, 1)
        all_frame=torch.transpose(all_frame, 0, 1)
        state          = torch.zeros(batch_size, self.rnn_size).to(device)

        # Encoding
        for i in range(self.source_seq_len+self.target_seq_len-1):
            # Apply the RNN cell
            state = self.cell(all_frame[i], state)
            # Apply dropout in training
            state = F.dropout(state, self.dropout, training=self.training)

        outputs = []
        prev    = None
    # Decoding, sequentially
        for i, inp in enumerate(all_frame):
            # Use teacher forcing?
            if prev is not None:
                inp = loop_function(prev, i)
            #inp = inp.detach()

            state  = self.cell(inp, state)
            # Output is seen as a residual to the previous value
            output = inp + self.fc1(F.dropout(state,self.dropout,training=self.training))
            outputs.append(output.view([1, batch_size, self.input_size]))
            prev = output
        outputs = torch.cat(outputs, 0)
        outputs=torch.transpose(outputs, 0, 1)

        base = torch.flatten(outputs, start_dim=1)

        output_lineal = self.fc2(base)
        output_prob=self.fc3(output_lineal)
        # torch.transpose(outputs, 0, 1)
        # Size should be batch_size x target_seq_len x input_size
        return output_prob