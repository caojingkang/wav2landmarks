import os

from numpy import random
from tensorboardX import SummaryWriter

def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory, mode=0o775)
        logger = Wav2EdgeLogger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger



class Wav2EdgeLogger(SummaryWriter):
    def __init__(self, logdir):
        super(Wav2EdgeLogger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
