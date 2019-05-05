import torch

class Visualizer:
    def __init__(self, log_dir):
        pass

    def draw(self, target_seq: torch.Tensor, pred_seq: torch.Tensor):
        assert len(target_seq) == len(pred_seq), 'the seq of two must be same'
        assert target_seq.shape[1:] == (68, 2) and pred_seq.shape[1:] == (68, 2)
