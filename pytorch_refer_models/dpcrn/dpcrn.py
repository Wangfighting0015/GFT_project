# coding: utf-8
# Author：WangTianRui
# Date ：2021-05-21 10:19
import math
from utils.stft import *
import dpcrn.dprnn_block as dprnn_block


def complex_cat(x1, x2):
    x1_real, x1_imag = torch.chunk(x1, 2, dim=1)
    x2_real, x2_imag = torch.chunk(x2, 2, dim=1)
    return torch.cat(
        [torch.cat([x1_real, x2_real], dim=1), torch.cat([x1_imag, x2_imag], dim=1)], dim=1
    )


class CausalConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(CausalConv, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.left_pad = kernel_size[1] - 1
        # padding = (kernel_size[0] // 2, 0)
        padding = (kernel_size[0] // 2, self.left_pad)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding)

    def forward(self, x):
        """
        :param x: B,C,F,T
        :return:
        """
        B, C, F, T = x.size()
        # x = F.pad(x, [self.left_pad, 0])
        return self.conv(x)[..., :T]


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, output_padding):
        super(CausalTransConvBlock, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                                             stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        """
        因果反卷积
        :param x: B,C,F,T
        :return:
        """
        T = x.size(-1)
        conv_out = self.trans_conv(x)[..., :T]
        return conv_out


class DPCRN(nn.Module):
    def __init__(self, rnn_hidden=128, win_len=512, hop_len=128, fft_len=512, win_type='hanning',
                 kernel_num=(32, 32, 32, 64, 128)):
        super(DPCRN, self).__init__()
        self.rnn_hidden = rnn_hidden

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.win_type = win_type

        self.kernel_size = ([5, 2], [3, 2], [3, 2], [3, 2], [3, 2])
        self.strides = ([2, 1], [2, 1], [1, 1], [1, 1], [1, 1])
        self.kernel_num = (2,) + kernel_num

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    CausalConv(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=self.kernel_size[idx],
                        stride=self.strides[idx]
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
        hidden_dim = 128
        self.enhance = nn.Sequential(
            dprnn_block.DPRnn(input_ch=kernel_num[-1], F_dim=hidden_dim, hidden_ch=kernel_num[-1]),
            dprnn_block.DPRnn(input_ch=kernel_num[-1], F_dim=hidden_dim, hidden_ch=kernel_num[-1])
        )
        self.transform = nn.Linear(self.rnn_hidden, hidden_dim * self.kernel_num[-1])
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        CausalTransConvBlock(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=self.kernel_size[idx - 1],
                            stride=self.strides[idx - 1],
                            padding=(self.kernel_size[idx - 1][0] // 2, 0),
                            output_padding=(self.strides[idx - 1][0] - 1, 0)
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1]),
                        nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        CausalTransConvBlock(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=self.kernel_size[idx - 1],
                            stride=self.strides[idx - 1],
                            padding=(self.kernel_size[idx - 1][0] // 2, 0),
                            output_padding=(self.strides[idx - 1][0] - 1, 0)
                        )
                    )
                )
        # self.ln_in = nn.LayerNorm([2, 200])
        self.gft = ConvSTFT(512, self.hop_len, self.fft_len, "hanning", 'complex')
        self.igft = ConviSTFT(512, self.hop_len, self.fft_len, "hanning", 'complex')
    def forward(self, x):
        real, imag = stft_splitter(x, self.gft, fft_len=self.fft_len, hop_len=self.hop_len)
        # real = stft[:, :self.fft_len // 2 + 1]
        # imag = stft[:, self.fft_len // 2 + 1:]
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        spec_phase = torch.atan(imag / (real + 1e-8))
        phase_adjust = (real < 0).to(torch.int) * torch.sign(imag) * math.pi
        spec_phase = spec_phase + phase_adjust
        spec_complex = torch.stack([real, imag], dim=1)[:, :, 1:]  # B,2,256,T

        out = spec_complex
        encoder_out = []
        for idx, encoder in enumerate(self.encoder):
            out = encoder(out)
            #print(out.shape())
            encoder_out.append(out)
            # print(out.size())

        # B, C, D, T = out.size()
        # print(out.size())
        out = self.enhance(out)
        # print(out.size())

        for idx in range(len(self.decoder)):
            out = torch.cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        #mask_real = F.pad(mask_real, [0, 0, 1, 0], value=1e-8)
        #mask_imag = F.pad(mask_imag, [0, 0, 1, 0], value=1e-8)
        mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        real_phase = mask_real / (mask_mags + 1e-8)
        imag_phase = mask_imag / (mask_mags + 1e-8)
        mask_phase = torch.atan(
            imag_phase / (real_phase + 1e-8)
        )
        phase_adjust = (real_phase < 0).to(torch.int) * torch.sign(imag_phase) * math.pi
        mask_phase = mask_phase + phase_adjust
        mask_mags = torch.tanh(mask_mags)  # mask 所以要tanh
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        real = est_mags * torch.cos(est_phase)
        imag = est_mags * torch.sin(est_phase)
        result=stft_mixer(real, imag, self.igft, fft_len=self.fft_len, hop_len=self.hop_len)
        
        #return stft_mixer(real, imag, n_fft=self.fft_len, hop_len=self.hop_len)
        return result

