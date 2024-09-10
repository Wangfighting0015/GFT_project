import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'



import time
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import AGNNConv


#from utils.gsp import *
from utils.stft import *

class Encoder(nn.Module):
    """
    Class of upsample block
    """

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45, padding=(0, 0)):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)

        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.PReLU = nn.PReLU()

    def forward(self, x):
        conved = self.conv(x)
        normed = self.bn(conved)
        acted = self.PReLU(normed)

        return acted


class Decoder(nn.Module):
    """
    Class of downsample block
    """

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45,
                 output_padding=(0, 0), padding=(0, 0), last_layer=False):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding

        self.last_layer = last_layer

        self.convt = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.filter_size, stride=self.stride_size,
                                       output_padding=self.output_padding, padding=self.padding)

        if not self.last_layer:
            self.bn = nn.BatchNorm2d(num_features=self.out_channels)
            self.PReLU = nn.PReLU()

    def forward(self, x):

        conved = self.convt(x)

        if not self.last_layer:
            normed = self.bn(conved)
            output = self.PReLU(normed)
        else:
            output = conved

        return output


class GUNET(nn.Module):
    def __init__(self, device='cuda',
                 rnn_hidden=1024,
                 nfft=512, win_l=None, hop_len=128, matrix='L',
                 kernel_size=5,
                 kernel_num=(16, 32, 64, 128, 128, 128)
                 ):
        super(GUNET, self).__init__()
        self.rnn_hidden = rnn_hidden

        self.kernel_size = kernel_size
        self.kernel_num = (1,) + kernel_num

        self.nfft = nfft
        self.hop_len= hop_len
        self.device = device
        #self.gsp = GSP(nfft=nfft, win_l=win_l, hop_len=hop_len, device=device, matrix=matrix, gnn=True)

        self.encoderGCN = AGNNConv()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(Encoder(in_channels=self.kernel_num[idx], out_channels=self.kernel_num[idx + 1],
                                        filter_size=(self.kernel_size, 2), stride_size=(2, 1) , padding=(2, 0)))

        hidden_dim = int(self.nfft / (2 ** (len(self.kernel_num) - 1)) )

        self.enhance = nn.LSTM(
            input_size=hidden_dim * self.kernel_num[-1],
            hidden_size=self.rnn_hidden,
            num_layers=2,
            dropout=0.0,
            batch_first=False
        )

        for idx in range(len(self.kernel_num) - 1, 0, -1):
            last_layer = False if idx != 1 else True
            self.decoder.append(Decoder(in_channels=self.kernel_num[idx] * 2, out_channels=self.kernel_num[idx - 1],
                                        filter_size=(self.kernel_size, 2), stride_size=(2, 1) , padding=(2, 0),
                                        output_padding=(1, 0), last_layer=last_layer))

        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()
 
 
        self.gft = ConvSTFT(512, self.hop_len, self.nfft, "hanning", 'complex')
        self.igft = ConviSTFT(512, self.hop_len, self.nfft, "hanning", 'complex')
    def forward(self, x):
    
        #gft = self.gsp.ST_GFT(x)
        #out = gft.unsqueeze(1)
        #print(x.shape)
        #N=x.shape[1]
        gft, imag = stft_splitter(x, self.gft, fft_len=self.nfft, hop_len=self.hop_len)
        out = gft.unsqueeze(1)


        # encoder
        encoder_out = []
        for idx, encoder in enumerate(self.encoder):
            out = F.pad(out, [1, 0, 0, 0])
            out = encoder(out)
            encoder_out.append(out)

        # LSTM
        B, C, D, T = out.size()
        out = out.permute(3, 0, 1, 2)
        out = torch.reshape(out, [T, B, C * D])
        out, _ = self.enhance(out)
        out = torch.reshape(out, [T, B, C, D])
        out = out.permute(1, 2, 3, 0)

        # decoder
        for idx in range(len(self.decoder)):
            out = torch.cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
            out = out[..., 1:]

        mask = out.squeeze()
        mask = torch.tanh(mask)
        enhance = mask * gft

        out_wav=stft_mixer(enhance, imag, self.igft, fft_len=self.nfft, hop_len=self.hop_len)
        #print(x.shape[1])
        #out_wav = out_wav[:, :x.shape[1]]
        #print(out_wav.shape)
        #out_wav = self.gsp.iST_GFT(enhance)
        #out_wav = out_wav[:, :x.shape[1]]
        # out_wav = torch.clamp_(out_wav, -1, 1)
        #return out_wav, torch.sign(enhance)
        return out_wav