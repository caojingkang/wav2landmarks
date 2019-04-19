import numpy as np
import torch
import cv2
from numpy import linalg


NUM_LANDMARKS = 68
TWICE_LANDMARKS = NUM_LANDMARKS * 2


def l2distance(points, center=0):
    # assert points.shape[-1] == center.shape[-1]
    points = points
    distances = linalg.norm(points, axis=-1)
    return distances


def encode_landmarks(landmarks: np.ndarray):
    assert landmarks.shape == (NUM_LANDMARKS, 2)
    center = np.mean(landmarks, axis=-2)
    # print('center is {}'.format(center))
    landmarks = landmarks - center
    distances = l2distance(landmarks)
    max_distance = np.max(distances)
    landmarks = landmarks / max_distance
    landmarks = landmarks.reshape(TWICE_LANDMARKS)
    return landmarks


def encode_landmarks_seq(landmarks_seq, seq_len):
    '''
    :param landmarks_seq:  seq * 68 * 2
    :param seq_len:
    :return:
    '''
    # 68 * 2 to 136 for linear interpolation
    assert landmarks_seq.shape[1:] == (NUM_LANDMARKS, 2)
    # for i in range(len(landmarks_seq)):
    #     landmarks_seq[i] = encode_landmarks(landmarks_seq[i])
    landmarks_seq = np.array(list(map(encode_landmarks,
                                      landmarks_seq)))
    landmarks_seq = landmarks_seq.astype('float32')
    landmarks_seq = cv2.resize(landmarks_seq, dsize=(TWICE_LANDMARKS, seq_len),
                               interpolation=cv2.INTER_LINEAR)
    # make sure that all num between [-0.5, 0.5] for tahn funciton
    return landmarks_seq / 2

def to_gpu(x: torch.Tensor):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)




if __name__ == '__main__':
    a = np.arange(24).reshape(4, 3, 2)
    print(a.reshape(-1, 6))
    seq = encode_landmarks_seq(a, 7)
    print(seq)
