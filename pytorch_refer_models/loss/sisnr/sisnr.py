# coding: utf-8
# Author：WangTianRui
# Date ：2021-08-19 14:21
import torch


def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, training=False, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10(target_norm / (noise_norm + eps) + eps)
    return torch.mean(snr)


def loss(inputs, label, training=False):
    return -(si_snr(inputs, label))


if __name__ == '__main__':
    test_inp = torch.tensor([0.1, 2., 3., 4.])
    test_ref = torch.tensor([5., 0.6, 7., 8.])
    print(loss(test_inp, test_ref))
