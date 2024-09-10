# coding: utf-8
# Author：WangTianRui
# Date ：2021-09-01 10:29
import warnings

warnings.filterwarnings("ignore")
import os, sys, json

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
import numpy as np
import torch
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import time, threading
from pesq import pesq
from pystoi import stoi
from pathlib import Path
from model import stft_splitter, stft_mixer, Network
from lava.lib.dl import slayer
import torch.nn as nn
import drawer

def get_all_wavs(root):
    files = []
    for p in Path(root).iterdir():
        if str(p).endswith(".wav"):
            files.append(str(p))
        for s in p.rglob('*.wav'):
            files.append(str(s))
    return list(set(files))


def audiowrite(destpath, audio, sample_rate=16000):
    '''Function to write audio'''

    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)
    sf.write(destpath, audio, sample_rate)
    return


def get_pesq(clean, estimate, rate, mode="wb"):
    return pesq(rate, ref=clean, deg=estimate, mode=mode)


def get_stoi(clean, estimate, rate):
    return stoi(clean, estimate, rate, extended=False)


def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_sdr(est, clean, eps=1e-8):
    s1_s2_norm = l2_norm(est, clean)
    s2_s2_norm = l2_norm(clean, clean)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * clean
    e_nosie = est - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10(target_norm / (noise_norm + eps) + eps)
    return torch.mean(snr)
EPS=1e-9
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

class Evaler:
    def __init__(self, root_home, csv_name, model, num_thread=5):
        super(Evaler, self).__init__()
        self.csv_name = csv_name
        self.pd_dict = {"noisy": [], "pesq_wb": [], "pesq_nb": [], "si-sdr": [], "stoi": [], "time": []}
        self.root_home = root_home
        self.clean_home = os.path.join(self.root_home, "clean")
        self.noisy_home = os.path.join(self.root_home, "noisy")
        self.noisy_pathes = get_all_wavs(self.noisy_home)
        self.num_thread = num_thread
        self.model = model
        self.device = next(self.model.parameters()).device
        self.ok_flag = np.zeros(self.num_thread)
        if not os.path.exists("./dns_csvs"):
            os.mkdir("./dns_csvs")
        self.csv_save_path = os.path.join("./dns_csvs", "%s.csv" % csv_name)

    def eval(self):
        for index in tqdm(range(len(self.noisy_pathes))):
            noisy_path = self.noisy_pathes[index]
            noisy, fs = sf.read(noisy_path, dtype="float32")
            noisy_name = os.path.basename(noisy_path)

            id_ = noisy_name.split("_fileid_")[1].split("_")[0]
            clean_path = os.path.join(self.clean_home, "clean_fileid_%s" % id_)

            clean, fs = sf.read(clean_path, dtype="float32")
            res = len(noisy) % self.model.hop_len
            if res != 0:
                noisy = np.pad(noisy, (0, self.model.hop_len - res), "constant", constant_values=(1e-8, 1e-8))

            with torch.no_grad():
                net_inp = torch.tensor(noisy)[None].to(self.device)
                start_time = time.time()
                estimate = self.model(net_inp)
                cost_itme = time.time() - start_time
                estimate = estimate.cpu().numpy().flatten()
            if res != 0:
                estimate = estimate[:-(self.model.hop_len - res)]

            pesq_wb = get_pesq(clean=clean, estimate=estimate, rate=fs, mode="wb")
            pesq_nb = get_pesq(clean=clean, estimate=estimate, rate=fs, mode="nb")
            stoi_score = get_stoi(clean=clean, estimate=estimate, rate=fs)
            # SI_SDR = si_sdr(est=torch.tensor(estimate), clean=torch.tensor(clean)).item()
            SI_SDR = si_snr(torch.tensor(clean), torch.tensor(estimate)).item()

            self.pd_dict["noisy"].append(noisy_name)
            self.pd_dict["pesq_wb"].append(pesq_wb)
            self.pd_dict["pesq_nb"].append(pesq_nb)
            self.pd_dict["stoi"].append(stoi_score)
            self.pd_dict["si-sdr"].append(SI_SDR)
            self.pd_dict["time"].append(cost_itme)
        
        print("all ok....")
        df = pd.DataFrame(self.pd_dict)
        descri = df.describe()
        print(self.csv_save_path)
        print(descri)
        df.to_csv(self.csv_save_path, encoding='utf_8_sig', index=False)


def load_best_param(log_file, model, index, gpu=False, test=True):
    print(model)
    noisy_abs, noisy_arg = stft_splitter(torch.randn(1, 16000))
    denoised_abs = model(noisy_abs)
    model_path = os.path.join(log_file, "_ckpt_epoch_%d.ckpt" % index)
    if not os.path.exists(model_path):
        exit("log path error:" + model_path)
    if gpu:
        ckpt = torch.load(model_path, map_location=torch.device("cuda:0"))
    else:
        ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt)
    # if test:
    #     model.eval()
    print("load param ok", model_path)
    return model

class ModelSystem(nn.Module):
    def __init__(self, denoiser, n_fft=512, hop_len=128):
        super(ModelSystem, self).__init__()
        self.encoder = stft_splitter
        self.decoder = stft_mixer
        self.model = denoiser
        self.n_fft = n_fft
        self.hop_len = hop_len

    def forward(self, noisy, out_delay=0):
        noisy_abs, noisy_arg = self.encoder(noisy, n_fft=self.n_fft, hop_len=self.hop_len)
        denoised_abs = self.model(noisy_abs)
        noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
        est_wave = self.decoder(denoised_abs, noisy_arg, n_fft=self.n_fft, hop_len=self.hop_len)
        return est_wave

if __name__ == '__main__':
    infos = [Network(), "/home/wang/codes/py/XTeam/baseline_solution/sdnn/exp/", 60, "baseline"]
    print(infos[1:])
    
    model_ = ModelSystem(load_best_param(log_file=infos[1], model=infos[0], index=infos[2], gpu=False))
    evaler = Evaler(root_home=r"/home/wang/codes/py/XTeam/no_reverb",
                    csv_name=infos[3], model=model_, num_thread=1)
    evaler.eval()
