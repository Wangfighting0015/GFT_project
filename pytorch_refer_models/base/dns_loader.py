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

import numpy as np
import os, torch, random, soundfile, math
from numpy.linalg import norm
from torch.utils.data import Dataset, DataLoader


seed = 102
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)


class DNSDataset(data.Dataset):
    def __init__(self, json_dir, data_home, audio_len=None, valid=False, only_two=True):
        super(DNSDataset, self).__init__()
        self.json_dir = json_dir
        self.data_home = data_home
        with open(json_dir, "r") as f:
            self.mix_infos = json.load(f)
        self.wav_ids = list(self.mix_infos.keys())
        self.only_two = only_two
        self.audio_len = audio_len
        self.valid = valid

    def __len__(self):
        return len(self.wav_ids)

    def __getitem__(self, item):
        utt_info = self.mix_infos[self.wav_ids[item]]
        temp = os.path.join(self.data_home, utt_info["mix"])
        assert os.path.exists(temp), temp
        noisy = sf.read(os.path.join(self.data_home, utt_info["mix"]), dtype="float32")[0]
        clean_path = os.path.join(self.data_home, utt_info["clean"])
        clean = sf.read(clean_path, dtype="float32")[0]
        print(len(clean))

        #if len(noisy) == len(clean) and len(noisy) != self.audio_len:
        if len(clean) <= self.audio_len:
           shortage = self.audio_len - len(clean)
           print(shortage)
           clean = np.pad(clean, (0, shortage), 'constant')  # 用0填充
           noisy = np.pad(noisy, (0, shortage), 'constant')  # 用0填充
        random_start = np.random.randint(low=0, high=len(clean)-self.audio_len-1)
            #random_start= np.int64(random.random() * (abs(clean.shape[0] - self.audio_len)))
        noisy = noisy[random_start:random_start+self.audio_len]
        clean = clean[random_start:random_start+self.audio_len]
        mask = np.zeros(1)

        noisy, clean = torch.from_numpy(noisy), torch.from_numpy(clean)
        return noisy, clean, mask

def load_wav(path, wav_file):
    #print(os.path.join(path, wav_file))
    wav, sr = soundfile.read(os.path.join(path, wav_file))
    wav = wav.astype('float32')
    # 能量归一化
    # wav = wav / ((np.sqrt(np.sum(wav ** 2)) / (wav.size + 1e-7)) + 1e-7)
    return wav


class VCTKTrain(Dataset):
    def __init__(self, dataset_path, train_files, wav_dur=3, is_trian=True):
        super(VCTKTrain, self).__init__()
        train_files = np.loadtxt(train_files, dtype='str').tolist()
        if is_trian:
            self.train_files = train_files[:10415]
        else:
            self.train_files = train_files[10415:]

        self.dataset_path = dataset_path
        self.max_len = wav_dur * 16000

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, index):
        # 读取干净语音
        clean_wav = load_wav(os.path.join(self.dataset_path, 'clean_trainset_28spk_wav'),
                             self.train_files[index] + '.wav')
        noisy_wav = load_wav(os.path.join(self.dataset_path, 'noisy_trainset_28spk_wav'),
                             self.train_files[index] + '.wav')
                             
        #print(clean_wav.shape[0])
        # 裁剪至固定长度
        clean_wav, noisy_wav = self.cut(clean_wav, noisy_wav)
        mask = np.zeros(1)
        noisy_wav, clean_wav = torch.from_numpy(noisy_wav), torch.from_numpy(clean_wav)
        # soundfile.write(f'./output/CLEAN.wav', clean_wav.astype('int16'), 16000)
        # soundfile.write(f'./output/NOISY.wav', noisy_wav.astype('int16'), 16000)
        return noisy_wav, clean_wav, mask

    def cut(self, clean_wav, noisy_wav):
        # 用0填充，保持每个batch中的长度一致
        if clean_wav.shape[0] <= self.max_len:
            shortage = self.max_len - clean_wav.shape[0]
            clean_wav = np.pad(clean_wav, (0, shortage), 'constant')  # 用0填充
            noisy_wav = np.pad(noisy_wav, (0, shortage), 'constant')  # 用0填充

        start = np.int64(random.random() * (clean_wav.shape[0] - self.max_len))
        return clean_wav[start: start + self.max_len], noisy_wav[start: start + self.max_len]


class VCTKEval(Dataset):
    def __init__(self, dataset_path, test_files):
        super(VCTKEval, self).__init__()
        self.test_files = np.loadtxt(test_files, dtype='str')
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, index):
        clean_wav = load_wav(os.path.join(self.dataset_path, 'clean_testset_wav'),
                             self.test_files[index] + '.wav')
        noisy_wav = load_wav(os.path.join(self.dataset_path, 'noisy_testset_wav'),
                             self.test_files[index] + '.wav')
        return torch.from_numpy(noisy_wav), torch.from_numpy(clean_wav)