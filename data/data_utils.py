import random
import numpy as np
import torch
import torch.utils.data


class MelLandmarkCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, opt):
        pass

    def __call__(self, batch):
        """Collate's training batch from normalized  mel-spectrogram and
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        dim_mel = batch[0][0].size(0)
        mel_padded = torch.LongTensor(len(batch), dim_mel, max_input_len)
        mel_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][0]
            mel_padded[i, :, :mel.size(1)] = mel

        # Right zero-pad mel-spec
        dim_landmarks = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        # if max_target_len % self.n_frames_per_step != 0:
        #     max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
        #     assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        landmarks_padded = torch.FloatTensor(len(batch), dim_landmarks, max_target_len)
        landmarks_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            landmarks = batch[ids_sorted_decreasing[i]][1]
            landmarks_padded[i, :, :landmarks.size(1)] = landmarks
            gate_padded[i, landmarks.size(1) - 1:] = 1
            output_lengths[i] = landmarks.size(1)

        return mel_padded, input_lengths, landmarks_padded, \
               gate_padded, output_lengths
