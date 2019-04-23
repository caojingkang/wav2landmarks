from torch import nn
from torch.nn.modules import MSELoss


class Wav2EdgeLoss(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(Wav2EdgeLoss, self).__init__()
        self.loss = MSELoss(size_average, reduce, reduction)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)
