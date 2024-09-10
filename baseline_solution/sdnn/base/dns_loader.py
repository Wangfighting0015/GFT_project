# coding: utf-8
# Author：WangTianRui
# Date ：2020/10/15 18:47
import json
import os, time, logging

logging.basicConfig(format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s',
                    level=logging.INFO)
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils import data

seed = 102
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)


class DNSDataset(data.Dataset):
    dataset_name = "DNS"
    def __init__(self, json_dir, data_home, audio_len=None, only_two=True, valid=False):
        super(DNSDataset, self).__init__()
        self.json_dir = json_dir
        self.data_home = data_home
        with open(json_dir, "r") as f:
            self.mix_infos = json.load(f)
        self.wav_ids = list(self.mix_infos.keys())
        if isinstance(self.data_home, list):
            np.random.shuffle(self.wav_ids)
        self.only_two = only_two
        self.valid = valid
        self.audio_len = audio_len

    def __len__(self):
        return len(self.wav_ids)

    def __getitem__(self, item):
        utt_info = self.mix_infos[self.wav_ids[item]]
        temp = os.path.join(self.data_home, utt_info["mix"])
        assert os.path.exists(temp), temp
        noisy = sf.read(os.path.join(self.data_home, utt_info["mix"]), dtype="float32")[0]
        clean_path = os.path.join(self.data_home, utt_info["clean"])
        clean = sf.read(clean_path, dtype="float32")[0]

        if len(noisy) == len(clean) and len(noisy) != self.audio_len:
            if not self.valid:
                random_start = np.random.randint(low=0, high=len(clean)-self.audio_len-1)
            else:
                random_start = (len(clean)-self.audio_len-1)//2
            noisy = noisy[random_start:random_start+self.audio_len]
            clean = clean[random_start:random_start+self.audio_len]
        mask = np.zeros(1)

        noisy, clean = torch.from_numpy(noisy), torch.from_numpy(clean)
        return noisy, clean, mask
