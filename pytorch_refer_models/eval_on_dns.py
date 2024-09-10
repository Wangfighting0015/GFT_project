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
import torch.nn as nn

import argparse
from nsnet.nsnet import NsNet2 
from dpcrn.dpcrn import DPCRN 
from dcrn.dcrn import DCRN 
from dccrn.dccrn import DCCRN 
from mtfaa.mtfaa import MTFAANet 
from gunet.gunet import GUNET  
from loss.sisnr.sisnr import loss as si_snr
import librosa as lib
import utils.drawer as drawer

model_dict = {
    "nsnet": NsNet2,
    "dpcrn": DPCRN,
    "dcrn": DCRN,
    "dccrn": DCCRN,
    "mtfaa": MTFAANet,
    "gunet": GUNET,
}

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
    def __init__(self, root_home, csv_path, model, num_thread=5, ckpt=""):
        super(Evaler, self).__init__()
        self.csv_path = csv_path
        self.ckpt = ckpt
        self.pd_dict = {"noisy": [], "pesq_wb": [], "pesq_nb": [], "si-sdr": [], "stoi": [], "time": []}
        self.root_home = root_home
        self.clean_home = os.path.join(self.root_home, "clean_testset_wav")
        self.noisy_home = os.path.join(self.root_home, "noisy_testset_wav")
        self.noisy_pathes = get_all_wavs(self.noisy_home)
        self.num_thread = num_thread
        self.model = model
        self.device = next(self.model.parameters()).device
        self.ok_flag = np.zeros(self.num_thread)

    def eval(self):
        for index in tqdm(range(len(self.noisy_pathes))):
            noisy_path = self.noisy_pathes[index]
            noisy, fs = sf.read(noisy_path, dtype="float32")
            noisy_name = os.path.basename(noisy_path)

            #id_ = noisy_name.split("_fileid_")[1].split("_")[0]
            #clean_path = os.path.join(self.clean_home, "clean_fileid_%s" % id_)
            
            id_ = noisy_name.split(".")[0]
            clean_path = os.path.join(self.clean_home, id_+".wav" )

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
        print(self.csv_path)
        with open(self.ckpt + "dns_eval.log", "w") as wf:
            print(descri, file=wf)
        df.to_csv(self.csv_path, encoding='utf_8_sig', index=False)


def load_best_param(model_path, model, gpu=False, test=True):
    print(model)
    if not os.path.exists(model_path):
        exit("log path error:" + model_path)
    if gpu:
        ckpt = torch.load(model_path, map_location=torch.device("cuda:0"))
    else:
        ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt)
    print("load param ok", model_path)
    return model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # 2. 添加命令行参数
    parser.add_argument('--dns-test', type=str, default=r"/data1/wtt_work/Generalmodel_graph/dataset/VoiceBank+DEMAND")
    parser.add_argument('--csv-save-path', type=str, default=r"/data1/wtt_work/Generalmodel_graph/A_propopsed_method/Voicebank+DEMAND/XTeam-master-GFT/pytorch_refer_models/_ckpt_epoch_41.ckptdns_eval.log")
    parser.add_argument('--model-name', type=str, default="gunet")
    parser.add_argument('--chptpath', type=str, default=r"/data1/wtt_work/Generalmodel_graph/A_propopsed_method/Voicebank+DEMAND/XTeam-master-GFT/pytorch_refer_models/exp/gunet_result-20240712/_ckpt_epoch_41.ckpt")
    args = parser.parse_args()
    print(args)
    model = model_dict[args.model_name]()
    model = load_best_param(args.chptpath, model)
    
    if torch.cuda.is_available():
        model = model.to(device="cuda:0")
    
    evaler = Evaler(root_home=args.dns_test, csv_path=args.csv_save_path, model=model, num_thread=1, ckpt=args.chptpath)
    evaler.eval()
