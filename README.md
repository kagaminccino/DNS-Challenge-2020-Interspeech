# DNS-Challenge-2020-Interspeech
Interspeech 2020 DNS Challenge

## Usage
```bash
CUDA_VISIBLE_DEVICES=1 python -W ignore trainer.py --gpus '0' --default_save_path saved --checkpoint_path saved/ckpt  --batch_size=32 --num_layers=6 --min_nb_epochs=200 --max_nb_epochs=300 --learning_rate=0.0005 --frame_size=96 --stride=48 --accumulate_grad_batches=1 --track_grad_norm=2 --loss l1 --e_bits 0 --m_bits 0
