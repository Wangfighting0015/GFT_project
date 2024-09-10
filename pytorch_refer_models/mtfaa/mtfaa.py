"""
Multi-scale temporal frequency axial attention neural network (MTFAA).

shmzhang@aslp-npu.org, 2022
"""

import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as tf
from typing import List

from mtfaa.tfcm import TFCM
from mtfaa.asa import ASA
from mtfaa.phase_encoder import PhaseEncoder
from mtfaa.f_sampling import FD, FU
from mtfaa.erb import Banks
#from mtfaa.stft import STFT
import utils.drawer as drawer
from utils.gftfunction import *

def parse_1dstr(sstr: str) -> List[int]:
    return list(map(int, sstr.split(",")))


def parse_2dstr(sstr: str) -> List[List[int]]:
    return [parse_1dstr(tok) for tok in sstr.split(";")]


eps = 1e-10


class MTFAANet(nn.Module):

    def __init__(self,
                 n_sig=1,
                 PEc=4,
                 Co="48,96,192",
                 O="1,1,1",
                 causal=True,
                 bottleneck_layer=2,
                 tfcm_layer=6,
                 mag_f_dim=3,
                 win_len=32 * 16,
                 win_hop=8 * 16,
                 nerb=512,
                 sr=16000,
                 win_type="hann",
                 ):
        super(MTFAANet, self).__init__()
        self.hop_len = win_hop
        self.win_len = win_len
        self.PE = PhaseEncoder(PEc, n_sig)
        # 32ms @ 48kHz
        
        #self.stft = STFT(win_len, win_hop, win_len, win_type)
        self.ERB = Banks(nerb, win_len, sr)
        # print(self.ERB.filter.size(), self.ERB.filter_inv.size())
        # drawer.plot_mesh(self.ERB.filter_inv.matmul(self.ERB.filter).data, "")
        # drawer.plot_mesh(self.ERB.filter.data, "filter")
        self.encoder_fd = nn.ModuleList()
        self.encoder_bn = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.decoder_fu = nn.ModuleList()
        self.decoder_bn = nn.ModuleList()
        C_en = [PEc // 2 * n_sig] + parse_1dstr(Co)
        C_de = [4] + parse_1dstr(Co)
        O = parse_1dstr(O)
        for idx in range(len(C_en) - 1):
            self.encoder_fd.append(
                FD(C_en[idx], C_en[idx + 1]),
            )
            self.encoder_bn.append(
                nn.Sequential(
                    TFCM(C_en[idx + 1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    ASA(C_en[idx + 1], causal=causal),
                )
            )

        for idx in range(bottleneck_layer):
            self.bottleneck.append(
                nn.Sequential(
                    TFCM(C_en[-1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    ASA(C_en[-1], causal=causal),
                )
            )

        for idx in range(len(C_de) - 1, 0, -1):
            self.decoder_fu.append(
                FU(C_de[idx], C_de[idx - 1], O=(O[idx - 1], 0)),
            )
            self.decoder_bn.append(
                nn.Sequential(
                    TFCM(C_de[idx - 1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    ASA(C_de[idx - 1], causal=causal),
                )
            )
        # MEA is causal, so mag_t_dim = 1.
        self.mag_mask = nn.Conv2d(
            4, mag_f_dim, kernel_size=(3, 1), padding=(1, 0))
        self.real_mask = nn.Conv2d(4, 1, kernel_size=(3, 1), padding=(1, 0))
        self.imag_mask = nn.Conv2d(4, 1, kernel_size=(3, 1), padding=(1, 0))
        kernel = th.eye(mag_f_dim)
        kernel = kernel.reshape(mag_f_dim, 1, mag_f_dim, 1)
        self.register_buffer('kernel', kernel)
        self.mag_f_dim = mag_f_dim
        self.fft_len=512
        self.gft = ConvSTFT(512, self.hop_len, self.win_len, "hanning", 'complex')
        self.igft = ConviSTFT(512, self.hop_len, self.win_len, "hanning", 'complex')
    def forward(self, sigs):
        """
        sigs: list [B N] of len(sigs)
        """
        cspecs = []
        # for sig in sigs:
        #     cspecs.append(self.stft.transform(sig))
        #print((stft_splitter(sigs, self.gft, self.fft_len, self.hop_len)).size(), 'xx')
        #cspecs.append(stft_splitter(x, self.gft, self.fft_len, self.hop_len))
        
        '''cspecs.append(self.stft.transform(sigs))
        #print((stft_splitter(sigs, self.gft, self.fft_len, self.hop_len)).size(), 'xx')
        cspecs.append(self.stft.transform(sigs))
        cspecs.append(self.stft.transform(sigs))'''
        
        cspecs.append(stft_splitter(sigs, self.gft, self.fft_len, self.hop_len))
        cspecs.append(stft_splitter(sigs, self.gft, self.fft_len, self.hop_len))
        cspecs.append(stft_splitter(sigs, self.gft, self.fft_len, self.hop_len))
        
        
        
        # D / E ?
        D_cspec = cspecs[0]
        #print(D_cspec.shape, 'xg')
        mag = th.norm(D_cspec, dim=1)
        pha = torch.atan2(D_cspec[:, -1, ...], D_cspec[:, 0, ...])
        
        phase_encoder_out = self.PE(cspecs)
        #print("phase_encoder_out:", phase_encoder_out.size())
        
        out = self.ERB.amp2bank(phase_encoder_out)
        #print("ERB_out:", out.size())
        encoder_out = []
        for idx in range(len(self.encoder_fd)):
            out = self.encoder_fd[idx](out)
            encoder_out.append(out)
            out = self.encoder_bn[idx](out)

        for idx in range(len(self.bottleneck)):
            out = self.bottleneck[idx](out)

        for idx in range(len(self.decoder_fu)):
            out = self.decoder_fu[idx](out, encoder_out[-1 - idx])
            out = self.decoder_bn[idx](out)
        out = self.ERB.bank2amp(out)
        #print("ERB_out1:", out.size())
        # stage 1
        mag_mask = self.mag_mask(out)
        #print(mag_mask.shape, 'xm')
        mag_pad = tf.pad(
            mag[:, None], [0, 0, (self.mag_f_dim - 1) // 2, (self.mag_f_dim - 1) // 2])
        mag = tf.conv2d(mag_pad, self.kernel)
        mag = mag * mag_mask.sigmoid()
        mag = mag.sum(dim=1)
        # stage 2
        real_mask = self.real_mask(out).squeeze(1)
        imag_mask = self.imag_mask(out).squeeze(1)

        mag_mask = th.sqrt(th.clamp(real_mask ** 2 + imag_mask ** 2, eps))
        pha_mask = th.atan2(imag_mask + eps, real_mask + eps)
        real = mag * mag_mask.tanh() * th.cos(pha + pha_mask)
        imag = mag * mag_mask.tanh() * th.sin(pha + pha_mask)
        #print(real.shape,'xx')
        out_wav=stft_mixer(real, imag, self.igft, self.fft_len, self.hop_len)
        #print(out_wav.shape, 'x2222')
        #return mag, th.stack([real, imag], dim=1), out_wav
        return out_wav
        


def test_nnet():

    # noise supression (microphone, )
    nnet = MTFAANet(n_sig=1)
    inp = th.randn(3, 48000)
    mag, cspec, wav = nnet([inp])
    print(mag.shape, cspec.shape, wav.shape)
    # echo cancellation (microphone, error, reference,)
    nnet = MTFAANet(n_sig=3)
    mag, cspec, wav = nnet([inp, inp, inp])
    print(mag.shape, cspec.shape, wav.shape)


def test_mac():
    from thop import profile, clever_format
    import torch as th
    import soundfile as sf
    nnet = MTFAANet(causal=False, n_sig=1)
    nnet = drawer.load_best_param(r"G:\datas\harformer_exp\model_save\MTFAANet_offline", nnet, 84)
    # nnet = drawer.load_best_param(r"G:\datas\harformer_exp\model_save\MTFAANet_Causal", nnet, 42)
    # hop=8ms, win=32ms@48KHz, process 1s.
    # inp = torch.tensor(
    #     [sf.read(
    #         r"G:\datas\test\fileid10_cleanBAC009S0657W0284_noiseuI44_PzWnCA_snr5_level-19.wav",
    #         dtype="float32")[0][:16000 * 2]]
    # )
    # print(nnet(inp).size())
    # inp = th.randn(1, 2, 769, 126)
    # macs, params = profile(nnet, inputs=([inp, inp, inp],), verbose=False)
    # macs, params = clever_format([macs, params], "%.3f")
    # print('macs: ', macs)
    # print('params: ', params)
    # from thop import profile, clever_format
    #
    inp = torch.randn(1, 16000)
    macs, params = profile(nnet, inputs=(inp,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('macs: ', macs)
    print('params: ', params)
    print(drawer.numParams(nnet))

    """
    macs:  2.395G
    params:  2.176M
    """


if __name__ == "__main__":
    # test_nnet()
    test_mac()
