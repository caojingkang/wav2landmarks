import os

import torch
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset

from model.stft import TacotronSTFT
from utils import encode_landmarks_seq


class MelLandmarksDataset(Dataset):
    def __init__(self, opt, mode='train', transform=None):
        assert mode == 'train' or mode == 'test'
        self.wavs_root = os.path.join(opt['data_root'], mode + '_img')
        self.keypoints_root = os.path.join(opt['data_root'], mode + '_keypoints')
        wavs_name = filter(lambda file_name: file_name.endswith('.wav'), os.listdir(self.wavs_root))
        self.names = list(map(lambda wav_name: '.'.join(wav_name.split('.')[:-1]), wavs_name))
        self.max_wav_value = opt['max_wav_value']
        self.stft = TacotronSTFT(
            opt['filter_length'], opt['hop_length'], opt['win_length'],
            opt['n_mel_channels'], opt['sampling_rate'], opt['mel_fmin'],
            opt['mel_fmax'])
        np.random.seed(opt['random_seed'])
        np.random.shuffle(self.names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.wavs_root, self.names[idx] + '.wav')
        keypoints_path = os.path.join(self.keypoints_root, self.names[idx] + '.npy')
        return self.get_mel_landmarks_pair(wav_path, keypoints_path)

    def get_mel(self, wav_path):
        sampling_rate, audio = wavfile.read(wav_path)
        # sampling_rate, audio = 16000, np.random.normal(size=(48000))
        # print('audio shape {} with sample rate {}'.format(audio.shape, sampling_rate))
        audio = torch.FloatTensor(audio.astype(np.float32))
        # if sampling_rate != self.stft.sampling_rate:
        #     raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate))

        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec

    def get_landmark(self, keypoints_path):
        return np.load(keypoints_path)

    def get_mel_landmarks_pair(self, wav_path, keypoints_path):
        mel = self.get_mel(wav_path) # n_mel_channels * seq_len
        mel_len = mel.shape[-1]
        landmarks_seq = self.get_landmark(keypoints_path)
        landmarks_seq = encode_landmarks_seq(landmarks_seq, mel_len).T
        landmarks_seq = torch.FloatTensor(landmarks_seq) # TWICE_LANDMARKS * seq_len
        return mel, landmarks_seq


if __name__ == '__main__':
    import json
    with open('../config.json') as f:
        opt = json.load(f)

    dataset = MelLandmarksDataset(opt, mode='test')
    print(dataset[1][0].shape)
    print(dataset[1][1].shape)
