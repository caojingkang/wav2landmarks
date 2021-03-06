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


def _encode_landmarks(landmarks: np.ndarray):
    assert landmarks.shape == (NUM_LANDMARKS, 2)
    center = np.mean(landmarks, axis=-2)
    # print('center is {}'.format(center))
    landmarks = landmarks - center
    distances = l2distance(landmarks)
    max_distance = np.max(distances)
    landmarks = landmarks / max_distance
    landmarks = landmarks.reshape(TWICE_LANDMARKS)
    return landmarks


def encode_landmarks_seq(landmarks_seq: np.ndarray, seq_len):
    """
    :param landmarks_seq:  seq * 68 * 2, as numpy.ndarray
    :param seq_len:
    :return: seq_len  *
    """
    # 68 * 2 to 136 for linear interpolation
    assert landmarks_seq.shape[1:] == (NUM_LANDMARKS, 2)
    # for i in range(len(landmarks_seq)):
    #     landmarks_seq[i] = encode_landmarks(landmarks_seq[i])
    landmarks_seq = np.array(list(map(_encode_landmarks,
                                      landmarks_seq)))
    landmarks_seq = landmarks_seq.astype('float32')
    # print('inside encoee seq; shape', landmarks_seq.shape)
    landmarks_seq = cv2.resize(landmarks_seq, dsize=(TWICE_LANDMARKS, seq_len),
                               interpolation=cv2.INTER_LINEAR)
    # make sure that all num between [-0.5, 0.5] for tahn funciton
    return landmarks_seq / 2


def _decode_landmarks(landmarks: torch.Tensor):
    assert landmarks.shape == (TWICE_LANDMARKS,)
    landmarks = landmarks.reshape(shape=(NUM_LANDMARKS, 2))
    # TODO decode the
    return landmarks


def decode_landmarks_seq(landmarks_seq: torch.Tensor, seq_len):
    """
    :param landmarks_seq: output from model, maybe in gpu
    :param seq_len:
    :return:  seq * 68 * 2
    """


def to_gpu(x: torch.Tensor):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def calculate_grad_norm(parameters, norm_type=2):
    """
    calculate the gradient norm of all model parameters.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)  # new a tensor in specific gpu
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


if __name__ == '__main__':
    a = np.arange(24).reshape(4, 3, 2)
    print(a.reshape(-1, 6))
    seq = encode_landmarks_seq(a, 7)
    print(seq)
