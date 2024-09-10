# coding: utf-8
# Author：WangTianRui
# Date ：2022/10/16 10:27
import torch.nn as nn
import torch


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, est_sig, clean_sig, training=False):
        return self.mse(est_sig, clean_sig)
