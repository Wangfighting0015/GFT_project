# coding: utf-8
# Author：WangTianRui
# Date ：2020/11/3 16:49
from utils.stft import *
from utils.complexnn import *
import os
import utils.drawer as drawer


class DCCRN(nn.Module):
    def __init__(self,
                 rnn_layer=2, rnn_hidden=256,
                 win_len=400, hop_len=128, fft_len=512, win_type='hanning',
                 use_clstm=True, use_cbn=False, masking_mode='E',
                 kernel_size=5, kernel_num=(32, 64, 128, 256, 256, 256)
                 ):
        super(DCCRN, self).__init__()
        self.rnn_layer = rnn_layer
        self.rnn_hidden = rnn_hidden

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.win_type = win_type

        self.use_clstm = use_clstm
        self.use_cbn = use_cbn
        self.masking_mode = masking_mode

        self.kernel_size = kernel_size
        self.kernel_num = (2,) + kernel_num

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                        self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
            # print(self.encoder[-1])
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if self.use_clstm:
            rnns = []
            for idx in range(rnn_layer):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=2*hidden_dim * self.kernel_num[-1] if idx == 0 else self.rnn_hidden,
                        hidden_size=self.rnn_hidden,
                        batch_first=False,
                        projection_dim=2*hidden_dim * self.kernel_num[-1] if idx == rnn_layer - 1 else None
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=2*hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_hidden,
                num_layers=2,
                dropout=0.0,
                batch_first=False
            )
            self.transform = nn.Linear(self.rnn_hidden, hidden_dim * self.kernel_num[-1])
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            self.kernel_num[idx - 1]),
                        nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        )
                    )
                )
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()
            
        self.gft = ConvSTFT(512, self.hop_len, self.fft_len, "hanning", 'complex')
        self.igft = ConviSTFT(512, self.hop_len, self.fft_len, "hanning", 'complex')
    def forward(self, x):
        real, imag = stft_splitter(x, self.gft, fft_len=self.fft_len, hop_len=self.hop_len)
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        spec_phase = torch.atan2(imag, real)
        spec_complex = torch.stack([real, imag], dim=1)[:, :, 1:]  # B,2,256

        out = spec_complex
        #print(out.shape, 'x')
        encoder_out = []
        for idx, encoder in enumerate(self.encoder):
            out = encoder(out)
            encoder_out.append(out)
        B, C, D, T = out.size()
        out = out.permute(3, 0, 1, 2)
        #print(out.shape, 'x1')
        if self.use_clstm:
            r_rnn_in = out[:, :, :C // 2, :]
            i_rnn_in = out[:, :, C // 2:, :]
            r_rnn_in = torch.reshape(r_rnn_in, [T, B, C // 2 * D])
            i_rnn_in = torch.reshape(i_rnn_in, [T, B, C // 2 * D])

            #print(r_rnn_in.shape, 'x2')
            #print(i_rnn_in.shape, 'x3')

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])
            #print(r_rnn_in.shape, 'x21')
            #print(i_rnn_in.shape, 'x31')

            r_rnn_in = torch.reshape(r_rnn_in, [T, B, C // 2, D])
            i_rnn_in = torch.reshape(i_rnn_in, [T, B, C // 2, D])
            #print(r_rnn_in.shape, 'x211')
            #print(i_rnn_in.shape, 'x311')
            out = torch.cat([r_rnn_in, i_rnn_in], 2)
            
        else:
            out = torch.reshape(out, [T, B, C * D])
            out, _ = self.enhance(out)
            #print(out.shape,'x22')
            out = self.transform(out)
            #print(out.shape,'x33')
            out = torch.reshape(out, [T, B, C, D])
        out = out.permute(1, 2, 3, 0)
        for idx in range(len(self.decoder)):
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
            out = out[..., 1:]
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        #mask_real = F.pad(mask_real, [0, 0, 1, 0], value=1e-8)
        #mask_imag = F.pad(mask_imag, [0, 0, 1, 0], value=1e-8)

        # drawer.plot_mesh(mask_real[0].data, "mask_real")
        # drawer.plot_mesh(mask_imag[0].data, "mask_imag")

        if self.masking_mode == 'E':
            mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
            real_phase = mask_real / (mask_mags + 1e-8)
            imag_phase = mask_imag / (mask_mags + 1e-8)
            mask_phase = torch.atan2(
                imag_phase,
                real_phase
            )
            mask_mags = torch.tanh(mask_mags)  # mask 所以要tanh
            est_mags = mask_mags * spec_mags
            est_phase = spec_phase + mask_phase
            real = est_mags * torch.cos(est_phase)
            imag = est_mags * torch.sin(est_phase)
        elif self.masking_mode == 'C':
            real = real * mask_real - imag * mask_imag
            imag = real * mask_imag + imag * mask_real
        elif self.masking_mode == 'R':
            real = real * mask_real
            imag = imag * mask_imag

        #out_wav = stft_mixer(real, imag, fft_len=self.fft_len, hop_len=self.hop_len)
        out_wav=stft_mixer(real, imag, self.igft, fft_len=self.fft_len, hop_len=self.hop_len)
        
        #return stft_mixer(real, imag, n_fft=self.fft_len, hop_len=self.hop_len)
        return out_wav

