# coding: utf-8
# Author：WangTianRui
# Date ：2021/3/24 9:09
from base.Solver import Solver
import time, torch, sys
import torch.nn as nn
import argparse
from model import stft_splitter, stft_mixer, Network
import torch.nn.functional as F
from lava.lib.dl import slayer
EPS = 1e-8

def si_snr(target, estimate) -> torch.tensor:
    if not torch.is_tensor(target):
        target: torch.tensor = torch.tensor(target)
    if not torch.is_tensor(estimate):
        estimate: torch.tensor = torch.tensor(estimate)

    # zero mean to ensure scale invariance
    s_target = target - torch.mean(target, dim=-1, keepdim=True)
    s_estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    # <s, s'> / ||s||**2 * s
    pair_wise_dot = torch.sum(s_target * s_estimate, dim=-1, keepdim=True)
    s_target_norm = torch.sum(s_target ** 2, dim=-1, keepdim=True)
    pair_wise_proj = pair_wise_dot * s_target / s_target_norm

    e_noise = s_estimate - pair_wise_proj

    pair_wise_sdr = torch.sum(pair_wise_proj ** 2,
                              dim=-1) / (torch.sum(e_noise ** 2,
                                                   dim=-1) + EPS)
    return 10 * torch.log10(pair_wise_sdr + EPS)


class ModelSystem(nn.Module):
    def __init__(self, denoiser, loss):
        super(ModelSystem, self).__init__()
        self.encoder = stft_splitter
        self.decoder = stft_mixer
        self.model = denoiser
        self.loss = loss

    def forward(self, noisy, clean, training, out_delay=0):
        noisy_abs, noisy_arg = self.encoder(noisy)
        clean_abs, clean_arg = self.encoder(clean)
        
        denoised_abs = self.model(noisy_abs)
        
        noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
        clean_abs = slayer.axon.delay(clean_abs, out_delay)
        clean = slayer.axon.delay(clean, 512 // 4 * out_delay)
        
        est_wave = self.decoder(denoised_abs, noisy_arg)
        
        loss = -self.loss(clean, est_wave)
        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # 2. 添加命令行参数
    parser.add_argument('--data-home', type=str, default="")
    parser.add_argument('--save-home', type=str, default="")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--valid-num', type=int, default=1)
    parser.add_argument('--pretrained-idx', type=int, default=-1)
    parser.add_argument('--audio-len', type=int, default=160000)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    print(args)
    solver = Solver(audio_len=args.audio_len, valid_num=args.valid_num, data_home=args.data_home, save_home=args.save_home, batch_size=args.batch_size)
    model = Network()
    print(model)
    criterion = si_snr
    # ------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # ------------------------
    # solver.init(model=model, criterion=criterion, model_system=system, gpus=[4, 5, 6, 7])
    # solver.init(model=model, criterion=criterion, model_system=system, gpus=[6, 7])
    # solver.init(model=model, criterion=criterion, model_system=system, gpus=[0, 1])
    solver.init(model=model, criterion=criterion, model_system=ModelSystem, gpus=[0], optimizer=optimizer)
    start = time.time()
    solver.train()
    print("cost:", (time.time() - start) / 3600)
