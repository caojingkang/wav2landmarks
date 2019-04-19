from torch.utils.data import DataLoader, DistributedSampler, BatchSampler

from .wav_landmark_dataset import MelLandmarksDataset
from .data_utils import MelLandmarkCollate


def prepare_dataloaders(opt):
    # Get data, data loaders and collate function ready
    trainset = MelLandmarksDataset(opt, mode='train')
    valset = MelLandmarksDataset(opt, mode='test')
    collate_fn = MelLandmarkCollate(opt)

    train_sampler = DistributedSampler(trainset) \
        if opt['distributed_run'] else None
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=opt['batch_size'], pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn

if __name__ == '__main__':
    pass
