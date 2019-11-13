import torch
from allennlp.modules.elmo_lstm import ElmoLstm


class ElmoLSTMFrozenStates(ElmoLstm):
    def forward(self, inputs: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            A Tensor of shape ``(batch_size, sequence_length, hidden_size)``.
        mask : ``torch.LongTensor``, required.
            A binary mask of shape ``(batch_size, sequence_length)`` representing the
            non-padded elements in each sequence in the batch.

        Returns
        -------
        A ``torch.Tensor`` of shape (num_layers, batch_size, sequence_length, hidden_size),
        where the num_layers dimension represents the LSTM output from that layer.
        """
        batch_size, total_sequence_length = mask.size()
        stacked_sequence_output, final_states, restoration_indices = self.sort_and_run_forward(
            self._lstm_forward, inputs, mask
        )
        # print("stacked_sequence_output:")
        # print(stacked_sequence_output)
        # print(stacked_sequence_output.shape)

        num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
        # Add back invalid rows which were removed in the call to sort_and_run_forward.
        if num_valid < batch_size:
            zeros = stacked_sequence_output.new_zeros(
                num_layers, batch_size - num_valid, returned_timesteps, encoder_dim
            )
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)

            # # The states also need to have invalid rows added back.
            new_states = []
            for state in final_states:
                state_dim = state.size(-1)
                zeros = state.new_zeros(num_layers, batch_size - num_valid, state_dim)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2StackEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - returned_timesteps
        if sequence_length_difference > 0:
            zeros = stacked_sequence_output.new_zeros(
                num_layers,
                batch_size,
                sequence_length_difference,
                stacked_sequence_output[0].size(-1),
            )
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 2)

        # we just comment the states update so the state will be the same
        self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        # Has shape (num_layers, batch_size, sequence_length, hidden_size)
        res = stacked_sequence_output.index_select(1, restoration_indices)
        # print("stacked_sequence_output(res):")
        # print(res)
        # print(res.shape)
        return res
