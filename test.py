import torch

from data.dataloader import prepare_dataloaders
from model.model import Wav2Edge
from torch import nn

if __name__ == '__main__':
    import json

    with open('/Users/jiananwei/Desktop/GAN/wav2edge/config.json') as f:
        opt = json.load(f)

    torch.manual_seed(opt['random_seed'])
    train_loader, _, _ = prepare_dataloaders(opt)
    loss_fn = nn.MSELoss()
    for batch in train_loader:
        # print(batch['landmarks_padded'].shape)
        # print(batch['mel_padded'].shape)

        wav2edge = Wav2Edge(opt)
        (mel_padded, inputs_length), (landmarks_padded, _) = wav2edge.parse_batch(batch)
        assert landmarks_padded.size(2) == mel_padded.size(2), "sequence length of input output not match"
        outputs = wav2edge(mel_padded, inputs_length)
        print('output this: ', outputs)
        print('with shape: ', outputs.shape)
        loss = loss_fn(landmarks_padded, outputs)
        print('loss is', loss)

        break
