import torch

from data.dataloader import prepare_dataloaders
from data.wav_landmark_dataset import WavLandmarksDataset
from model.model import Wav2Edge
from torch import nn

from train import load_model

if __name__ == '__main__':
    import json

    with open('./config.json') as f:
        opt = json.load(f)

    # dataset = WavLandmarksDataset(opt, mode='test')
    # print(dataset[1][0])
    # print(dataset[1][1][:, 0].shape==(136,))
    # exit(0)

    wav2edge = load_model(opt, '0')

    outputs = wav2edge.inference('/home/baby/wei/wav2edge/datasets/test_img/2019-03-02 14_23_23.wav')
    print(outputs.shape, outputs.device)
    exit(0)
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
