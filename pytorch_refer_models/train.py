# coding: utf-8
# Author：WangTianRui
# Date ：2021/3/24 9:09
from base.solver import Solver
import time, torch, sys
import torch.nn as nn
import argparse
from nsnet.nsnet import NsNet2 
from dpcrn.dpcrn import DPCRN 
from dcrn.dcrn import DCRN 
from dccrn.dccrn import DCCRN 
from mtfaa.mtfaa import MTFAANet
from gunet.gunet import GUNET  
from loss.sisnr.sisnr import loss as si_snr

model_dict = {
    "nsnet": NsNet2,
    "dpcrn": DPCRN,
    "dcrn": DCRN,
    "dccrn": DCCRN,
    "mtfaa": MTFAANet,
    "gunet": GUNET,
}

loss_dict = {
    "sisnr": si_snr,
}

class ModelSystem(nn.Module):
    def __init__(self, model, loss):
        super(ModelSystem, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, x, label, training):
        est = self.model(x)
        loss = self.loss(est, label, training=training)
        if not training:
            return loss,
        return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # 2. 添加命令行参数
    parser.add_argument('--data-home', type=str, default="/data1/wtt_work/Generalmodel_graph/dataset/VoiceBank+DEMAND")
    parser.add_argument('--save-home', type=str, default="/data1/wtt_work/Generalmodel_graph/A_propopsed_method/Voicebank+DEMAND/XTeam-master-GFT/pytorch_refer_models/exp/gunet_result-20240712")
    parser.add_argument('--model-name', type=str, default="gunet")
    parser.add_argument('--loss-name', type=str, default="sisnr")
    parser.add_argument('--batch-size', type=int, default="62")
    parser.add_argument('--load-param-index', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-descend-factor', type=float, default=0.75)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--stop-patience', type=int, default=20)
    args = parser.parse_args()
    print(args)
    solver = Solver(
        audio_len=int(3.0 * 16000), 
        valid_num=1, 
        data_home=args.data_home, 
        save_home=args.save_home, 
        batch_size=args.batch_size,
        load_param_index=args.load_param_index,
        lr=args.lr,
        lr_descend_factor=args.lr_descend_factor,
        stop_patience=args.stop_patience,
        patience=args.patience
    )
    model = model_dict[args.model_name]()
    # ------------------------
    criterion = loss_dict[args.loss_name]
    # ------------------------
    solver.init(model=model, criterion=criterion, model_system=ModelSystem, gpus=[5,6])
    start = time.time()
    solver.train()
    print("cost:", (time.time() - start) / 3600)
