import os
import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import rnn

from argparse import ArgumentParser
from multiprocessing import Pool

import pytorch_lightning as pl

from model import ConvBSRU
from datasets import DNS_Dataset
from optimizsers import RAdam
# from ranger import Ranger

from pesq import pesq
# from PMSQE.pmsqe_torch import PMSQE
# from pytorch_lamb import Lamb

def cal(x, y, l):
    return pesq(16000, y[:l], x[:l], 'nb')

def get_pesq(x, y, lens):
   
    y = list(y.squeeze(1).cpu().numpy())
    x = list(x.squeeze(1).cpu().numpy())
    lens = lens.tolist()

    with Pool(processes=16) as pool:
        pesqs = pool.starmap(
            cal, 
            iter([(deg, ref, l) for deg, ref, l in zip(x, y, lens)])
        )

    return torch.FloatTensor(pesqs).mean()

class DNS(pl.LightningModule):

    def __init__(self, hparams):
        super(DNS, self).__init__()
        self.hparams = hparams
        self.save = False
        if not os.path.exists(hparams.output_path):
            os.makedirs(hparams.output_path)
        
        '''
        if hparams.seed:
            seed = hparams.seed
            os.environ['PYTHONHASHSEED'] = str(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        '''

        self.net = ConvBSRU(
            frame_size=hparams.frame_size, 
            stride=hparams.stride,
            conv_channels=hparams.conv_channels, 
            num_layers=hparams.num_layers, 
            dropout=hparams.dropout, 
            rescale=hparams.rescale,
            bidirectional=hparams.bidirectional
        )

    def forward(self, x):
        return self.net(x)

    def loss(self, x, y, lens):
        mask = torch.zeros_like(x)
        
        for i, l in enumerate(lens):
            pad_len = self.hparams.stride - l % self.hparams.stride
            mask[i][0][:l] = 1.

        if self.hparams.loss == 'l1':
            return F.l1_loss(x*mask, y)
        
        elif self.hparams.loss == 'l2':
            return F.mse_loss(x, y)

    def training_step(self, batch, batch_idx):
        x, y, lens = batch
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y, lens)

        logger_logs = {'loss': loss}

        outputs = {
            'loss': loss,
            'progress_bar': {'loss': loss},
            'log': logger_logs
        }

        return outputs

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y, lens = batch
        y_hat = self.forward(x)

        val_loss = self.loss(y_hat, y, lens)
        val_pesq = get_pesq(y_hat, y, lens)

        # self.logger.experiment.add_scalar('val_loss', val_loss, batch_idx)
        # self.logger.experiment.add_scalar('val_pesq', val_pesq, batch_idx)

        outputs = {
            'val_loss': val_loss,
            'val_pesq': val_pesq
        }

        return outputs

    def validation_end(self, outputs):
        # OPTIONAL
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_pesq = torch.stack([x['val_pesq'] for x in outputs]).mean()
        if val_pesq > 2.1:
            self.save = True
        else:
            self.save = False

        pbar_logs = {
            'val_loss': val_loss,
            'val_pesq': val_pesq,
        }

        logger_logs = {
            'avg_val_loss': val_loss,
            'avg_val_pesq': val_pesq,
        }

        outputs = {
            'progress_bar': pbar_logs,
            'log': logger_logs
        }

        return outputs

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optimizer = RAdam(self.parameters(), lr=self.hparams.learning_rate)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # optimizer = Ranger(self.parameters(), lr=self.hparams.learning_rate)
        
        return optimizer

    def collate_fn(self, batch):
        x = rnn.pad_sequence([item[0] for item in batch]).transpose(0, 1)
        y = rnn.pad_sequence([item[1] for item in batch]).transpose(0, 1)
        lens = torch.LongTensor([item[2] for item in batch])
        return x, y, lens

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
                DNS_Dataset(self.hparams.data_dir, self.hparams.stride, train=True), 
                batch_size=self.hparams.batch_size, 
                collate_fn=self.collate_fn, 
                shuffle=True, 
                num_workers=4)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
                DNS_Dataset(self.hparams.data_dir, self.hparams.stride, train=False), 
                batch_size=self.hparams.batch_size, 
                collate_fn=self.collate_fn, 
                shuffle=False, 
                num_workers=4)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
                DNS_Dataset(self.hparams.data_dir, self.hparams.stride, train=False), 
                batch_size=self.hparams.batch_size, 
                collate_fn=self.collate_fn, 
                shuffle=False, 
                num_workers=4)

    # @pl.hooks.ModelHooks
    # def on_epoch_end(self):
    #     pass






    '''
    def on_save_checkpoint(self, checkpoint):
        if self.save:
            net = self.cpu()
            val_loader = self.val_dataloader()[0]
            data_list = val_loader.dataset.noisy_path
            data_list = [fname.split('/')[-1] for fname in data_list]
            enhanced = []
            self.eval()
            for i, (n, c, lens) in enumerate(val_loader):
                y_hat = self.forward(n)
                for j, y in enumerate(y_hat):
                    pad_len = self.hparams.stride - lens[j] % self.hparams.stride
                    torchaudio.save(
                        os.path.join(self.hparams.output_path, data_list[i * self.hparams.batch_size + j]), 
                        y[:, pad_len // 2: pad_len // 2 + lens[j]], 
                        16000
                    )
            self.cuda()
    '''
    def add_model_specific_args(parent_parser):
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.0002, type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--frame_size', default=96, type=int)
        parser.add_argument('--stride', default=48, type=int)
        parser.add_argument('--conv_channels', default=256, type=int)
        parser.add_argument('--num_layers', default=6, type=int)
        parser.add_argument('--dropout', default=0.0, type=float)
        parser.add_argument('--rescale', action='store_true')
        parser.add_argument('--bidirectional', action='store_true')
        parser.add_argument('--loss', default='l1', type=str)

        parser.add_argument('--data_dir', default='/Data/Dataset/DNS_Challenge', type=str)
        parser.add_argument('--seed', default=None, type=int)

        # training specific (for this model)
        parser.add_argument('--weights_summary', default='full', type=str)
        parser.add_argument('--max_nb_epochs', default=150, type=int)
        parser.add_argument('--min_nb_epochs', default=100, type=int)
        parser.add_argument('--default_save_path', default='/Data/Code/DNS-Challenge-2020-Interspeech', type=str)
        parser.add_argument('--output_path', default='/Data/Code/DNS-Challenge-2020-Interspeech/outputs', type=str)
        parser.add_argument('--checkpoint_path', default='/Data/Code/DNS-Challenge-2020-Interspeech/checkpoints', type=str)
        parser.add_argument('--gradient_clip_val', default=0.02, type=float)
        parser.add_argument('--track_grad_norm', default=-1, type=int)
        parser.add_argument('--log_save_interval', default=10, type=int)
        parser.add_argument('--accumulate_grad_batches', default=1, type=int)

        return parser


