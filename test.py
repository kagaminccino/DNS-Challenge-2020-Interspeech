import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils import rnn

import pytorch_lightning as pl

from datasets import DNS_Dataset
from tasks import DNS

output_path = '/Data/Code/DNS-Challenge-2020-Interspeech/outputs'

model = DNS.load_from_metrics(
    weights_path='./saved/ckpt/pesq_ckpt_epoch_195.ckpt',
    tags_csv='./saved/logs/version_0/meta_tags.csv',
)
model.cuda()
model.eval()
loader = model.val_dataloader()[0]
data_list = [fname.split('/')[-3:] for fname in loader.dataset.noisy_path]
# print(data_list)

for i, (n, c, l) in enumerate(loader):
    e = model.forward(n.cuda())
    for j, y in enumerate(e):
        pad_len = model.hparams.stride - l[j] % model.hparams.stride
        save_path = os.path.join(output_path, *data_list[i * model.hparams.batch_size + j])
        torchaudio.save(
            save_path,
            y[:, pad_len:pad_len + l[j]].cpu(), 
            16000
        )

