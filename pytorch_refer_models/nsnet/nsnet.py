# coding: utf-8
# Author：WangTianRui
# Date ：2020/11/22 19:20

import torch.nn as nn
import torch
import math
from utils.stft import *


class NsNet2(nn.Module):
    def __init__(self, nfft=512, hop_len=128):
        super(NsNet2, self).__init__()
        self.nfft = nfft
        self.hop_len = hop_len
        self.encoder = nn.Sequential(
            nn.Linear(512, 400),
            nn.PReLU(),
            nn.LayerNorm(400)
        )
        self.rnn = nn.Sequential(
            nn.GRU(input_size=400, hidden_size=400, batch_first=True, num_layers=2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(400, 600),
            nn.LayerNorm(600),
            nn.PReLU(),
            nn.Linear(600, 600),
            nn.LayerNorm(600),
            nn.PReLU(),
            nn.Linear(600, 512),
            nn.Sigmoid()
        )
        self.gft = ConvSTFT(512, self.hop_len, self.nfft, "hanning", 'complex')
        self.igft = ConviSTFT(512, self.hop_len, self.nfft, "hanning", 'complex')
    def forward(self, x):
        # x : B, T
        real, imag = stft_splitter(x, self.gft, fft_len=self.nfft, hop_len=self.hop_len)
        #real, imag = stft_splitter(x, n_fft=self.nfft, hop_len=self.hop_len)
        #print(real.shape, 'x')
        real = real.permute(0, 2, 1)
        #print(real.shape, 'x1')
        imag = imag.permute(0, 2, 1)
        log_pow = torch.log((real ** 2 + imag ** 2).clamp_(min=1e-12)) / torch.log(torch.tensor(10.0)) # B,F,T

        ff_result = self.encoder(log_pow)
        rnn_result, _ = self.rnn(ff_result)
        mask = self.decoder(rnn_result)

        real_result = (real * mask).permute(0, 2, 1)
        #print(real_result.shape, 'x2')
        imag_result = (imag * mask).permute(0, 2, 1)
        wav_out=stft_mixer(real_result, imag_result, self.igft, fft_len=self.nfft, hop_len=self.hop_len)
        #tt=stft_mixer(real_result, imag_result, n_fft=self.nfft, hop_len=self.hop_len)
        #print(tt.shape, 'x3')
        return wav_out
        #return stft_mixer(real_result, imag_result, n_fft=self.nfft, hop_len=self.hop_len)



