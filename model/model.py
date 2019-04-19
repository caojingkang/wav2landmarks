import torch
import torch.nn.functional as F
from torch import nn

from utils import to_gpu


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class Wav2Edge(nn.Module):
    def __init__(self, opt):
        super(Wav2Edge, self).__init__()
        assert opt['rnn_layers_num'] % 2 == 0
        self.rnn = nn.RNN(opt['n_mel_channels'], opt['rnn_hidden_dim'] // 2,
                          num_layers=opt['rnn_layers_num'], nonlinearity='tanh',
                          batch_first=True, bidirectional=True)
        self.linear_layer1 = LinearNorm(opt['rnn_hidden_dim'], opt['linear_dim'], w_init_gain='relu')
        self.linear_layer2 = LinearNorm(opt['linear_dim'], opt['n_landmarks_channels'], w_init_gain='tanh')

    def parse_batch(selfs, batch):
        mel_padded, input_lengths, landmarks_padded, \
        gate_padded, output_lengths = batch
        mel_padded = to_gpu(mel_padded).float()

        mel_padded = to_gpu(mel_padded).float()
        input_lengths = to_gpu(input_lengths).long()
        landmarks_padded = to_gpu(landmarks_padded)
        output_lengths = to_gpu(output_lengths).long()

        return (mel_padded, input_lengths), \
               (landmarks_padded, output_lengths)

    def forward(self, x: torch.Tensor, input_lengths):
        x = x.transpose(1, 2)  # batch x  time x channel
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        # self.rnn.flatten_parameters() #TODO
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True, padding_value=0.0)
        outputs = F.relu(self.linear_layer1(x))
        outputs = F.tanh(self.linear_layer2(outputs))

        outputs = outputs.transpose(1, 2)
        return outputs

    def inference(self):
        pass
