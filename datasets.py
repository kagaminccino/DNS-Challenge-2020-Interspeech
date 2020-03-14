import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import os
import glob
import random

class DNS_Dataset(Dataset):
    def __init__(self, data_dir, stride, train=True):
        self.data_dir = data_dir
        self.train = train
        if self.train:
            self.data_dir += '-223'
        self.tier = 'training' if train else 'datasets/test_set/synthetic/*'

        self.clean_root = os.path.join(self.data_dir, self.tier, 'clean')
        self.noisy_root = os.path.join(self.data_dir, self.tier, 'noisy')

        self.clean_path = self.get_path(self.clean_root)
        self.noisy_path = self.get_path(self.noisy_root)

        self.stride = stride

    def get_path(self, root):
        paths = glob.glob(os.path.join(root, '*.wav'))
        return sorted(paths, key=lambda x: int(x.split('_')[-1].replace('.wav', '')))

    def padding(self, x):
        x.unsqueeze_(0)
        len_x = x.size(-1)
        pad_len = self.stride - len_x % self.stride
        return F.pad(x, (pad_len, 0), mode='constant')

    def truncate(self, n, c):
        offset = 160000
        length = n.size(-1)
        start = torch.randint(length - offset, (1,))
        return n[:, start:start + offset], c[:, start:start + offset]

    def __len__(self):
        return len(self.noisy_path)

    def __getitem__(self, idx):
        clean = torchaudio.load(self.clean_path[idx])[0]
        noisy = torchaudio.load(self.noisy_path[idx])[0]
        if self.train:
            noisy, clean = self.truncate(noisy, clean)
        length = clean.size(-1)
        clean = self.padding(clean).squeeze(0)
        if self.train:
            noise = (torch.rand_like(clean) - 0.5) * 1e-3
            clean += noise
        noisy = self.padding(noisy).squeeze(0)

        # clean /= clean.abs().max()
        # noisy /= noisy.abs().max()

        return noisy, clean, length
