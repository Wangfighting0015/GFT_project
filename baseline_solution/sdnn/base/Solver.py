# coding: utf-8
# Author：WangTianRui
# Date ：2021/3/24 9:01
import warnings

warnings.filterwarnings("ignore")
import torch.nn as nn
import logging
import numpy as np
import json, gc
from base.dns_loader import DNSDataset
from torch.utils.data import DataLoader
import torch, time, os
from torch import optim

logging.basicConfig(format='%(asctime)s %(filename)s %(message)s', level=logging.INFO)
seed = 102
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)


class Solver(object):
    def __init__(self, audio_len=None, test=False, 
                 valid_num=3, data_home="", save_home="", batch_size=8,
                 load_pre_params=-1,
                 lr=1e-3, lr_descend_factor=0.5, stop_patience=20, patience=5):
        
        # load args parameters
        self.data_home = data_home
        self.save_home = save_home
        self.save_file = self.save_home
        self.checkpoint_dir = os.path.join(self.save_home, "checkpoints/")
        self.train_loader, self.val_loader = get_dns_data_loader(
            batch_size, 1,
            json_home=self.data_home, audio_len=audio_len,
            data_home=self.data_home,
        )
        self.load_param_index = load_pre_params
        self.model = None
        # ----------------------
        self.valid_losses_num = valid_num
        self.valid_losses = [[] for _ in range(self.valid_losses_num)]
        # ----------------------
        self.train_losses = []
        self.max_epoch = 999
        self.lr = lr
        self.optimizer = None
        self.lr_descend_factor = lr_descend_factor
        self.stop_patience = stop_patience
        self.saved_path = []
        self.criterion = None
        self.patience = None
        self.lr_descend_flag = 0
        self.early_stop_flag = 0
        self.patience = patience

    def init(self, model, criterion, gpus, model_system, optimizer=None):
        if gpus is None:
            logging.error("set your gpu codes like [0,1]")
            return
        os.environ['CUDA_VISIBLE_DEVICES'] = "".join([str(item) + "," for item in gpus])[:-1]
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        self.criterion = criterion
        os.makedirs(self.save_file, exist_ok=True)
        # Reset model
        if self.load_param_index >= 0:
            logging.info("loading param")
            model_path = os.path.join(self.save_file, "_ckpt_epoch_%d.ckpt" % self.load_param_index)
            if not os.path.exists(model_path):
                logging.error("model param path error:" + model_path)
                return
            else:
                ckpt = torch.load(model_path)
            model_state_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in ckpt.items() if k in model_state_dict and v.size()==model_state_dict[k].size()}
            print("pre load:%s"%(pretrained_dict.keys()))
            model_state_dict.update(pretrained_dict)
            model.load_state_dict(model_state_dict)
            logging.info("load param successful! " + str(self.load_param_index))
        self.model = model_system(model, criterion)
        self.model = nn.DataParallel(self.model).cuda()
        
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(),
                weight_decay=0,
                lr=self.lr
            )

    def train(self):
        logging.info("train:%d,valid:%d" % (len(self.train_loader), len(self.val_loader)))
        start_epoch = 0
        if self.load_param_index > 0:
            start_epoch = self.load_param_index
        for epoch in range(start_epoch, self.max_epoch):
            self.model.module.train()
            start = time.time()
            # logging.info('-' * 70)
            train_avg_loss = self._run_one_epoch(epoch, training=True)
            # Cross cv
            self.model.module.eval()
            valid_losses = self._run_one_epoch(epoch, training=False)
            self.make_epoch_log(epoch, start, train_avg_loss, valid_losses)
            # update valid loss
            if not self.valid_loss_update_and_check_better(valid_losses):
                logging.info("Not better")
            else:
                # save model each epoch
                model_save_path = os.path.join(
                    self.save_file, "_ckpt_epoch_%d.ckpt" % (epoch + 1)
                )
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                torch.save(self.model.module.model.state_dict(), model_save_path)
                self.saved_path.append(model_save_path)
                if len(self.saved_path) >= 20:
                    first_path = self.saved_path.pop(0)
                    if os.path.exists(first_path):
                        os.remove(first_path)
                logging.info('saving checkpoint model to %s' % model_save_path)

            if self.early_stop_flag >= self.stop_patience:
                self.make_final_log(start_epoch)
                logging.info("stop!")
                break

            if self.lr_descend_flag >= self.patience:
                last = self.optimizer.state_dict()['param_groups'][0]['lr']
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = last * self.lr_descend_factor
                print('\nLearning rate adjusted from %f to %f; patience:%d' % (
                    last, self.optimizer.state_dict()['param_groups'][0]['lr'], self.patience))
                self.make_final_log(start_epoch)
                self.lr_descend_flag = 0

    def _run_one_epoch(self, epoch, training=False):
        total_loss = 0
        valid_loss_total = [0 for _ in range(self.valid_losses_num)]
        data_loader = self.train_loader if training else self.val_loader
        batch_id = 0
        for batch_id, (x, y, mask) in enumerate(data_loader):
            if training:
                self.optimizer.zero_grad()
                if mask.sum().item() != 0:
                    batch_loss = self.model(x.cuda(), y.cuda(), training, mask=mask.cuda()).mean()
                else:
                    batch_loss = self.model(x.cuda(), y.cuda(), training).mean()
                if batch_loss.item() != batch_loss.item():
                    continue
                batch_loss.backward()
                total_loss += batch_loss.item()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    valid_losses = self.model(x.cuda(), y.cuda(), training)
                    for index in range(len(valid_loss_total)):
                        valid_loss_total[index] += valid_losses[index].mean().item()
        gc.collect()
        if training:
            return total_loss / (batch_id + 1)
        else:
            return np.array(valid_loss_total) / (batch_id + 1)

    def valid_loss_update_and_check_better(self, valid_losses):
        isbetter = False
        if len(self.valid_losses[0]) >= 1:
            for index in range(len(valid_losses)):
                if np.min(self.valid_losses[index]) > valid_losses[index]:
                    isbetter = True
                    self.lr_descend_flag = 0
                    self.early_stop_flag = 0
                    # self.valid_losses[index].append(valid_losses[index])
            if not isbetter:
                self.lr_descend_flag += 1
                self.early_stop_flag += 1
        else:
            isbetter = True
        for index in range(len(valid_losses)):
            self.valid_losses[index].append(valid_losses[index])
        return isbetter

    def make_epoch_log(self, epoch, start, train_avg_loss, losses):
        log = "epoch:%d|time: %.2f min|train_loss:%.4f|" % (int(epoch + 1), (time.time() - start) / 60, train_avg_loss)
        for index in range(len(losses)):
            log += "valid_loss_%d:%.4f;" % (index + 1, losses[index])
        logging.info(log)

    def make_final_log(self, start_epoch):
        log = ""
        for index in range(len(self.valid_losses)):
            log += "loss%d best:%d(%.5f)" % (index + 1,
                                             np.argmin(self.valid_losses[index]).item() + 1 + start_epoch,
                                             np.min(self.valid_losses[index]))
        log += "stop_patience:%d/%d\n" % (self.early_stop_flag, self.stop_patience)
        logging.info(log)


def load_best_param(log_file, model, gpu=False, test=True):
    """
    :param log_file: 保存参数的file
    :param model: 模型
    :param gpu: 是否使用GPU
    :return: 读取了最好参数的模型
    """
    if not os.path.exists(log_file):
        logging.error("log path error")
        return
    with open(os.path.join(log_file, "best_k_models.json"), "r") as f:
        best_k = json.load(f)
    best_model_path = min(best_k, key=best_k.get)
    logging.info("best model :", best_model_path)
    if not os.path.exists(best_model_path):
        ckpt_name = str(best_model_path).split('/')[-1]
        best_model_path = os.path.join(log_file, ckpt_name)
        if not os.path.exists(best_model_path):
            logging.error("model param path error")
            return
    if gpu:
        ckpt = torch.load(best_model_path, map_location=torch.device("cuda:0"))
    else:
        ckpt = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(ckpt)
    if test:
        model.eval()
    logging.info("load param ok")
    return model


def frame_and_freq(model_stft, audio_len, conf):
    test_inp = torch.randn(1, audio_len)
    conf["stft_aug"]["f"] = model_stft(test_inp)[0].size(1)
    conf["stft_aug"]["t"] = model_stft(test_inp)[0].size(0)
    return conf


# -------------------------------------------------------------------------------------


def get_dns_data_loader(batch_size, num_workers, json_home, data_home, audio_len=None):
    train_json_file = os.path.join(json_home, "train_file_info.json")
    val_json_file = os.path.join(json_home, "valid_file_info.json")
    train_set = DNSDataset(train_json_file, data_home=data_home, valid=False, audio_len=audio_len)
    if isinstance(data_home, list):
        val_set = DNSDataset(val_json_file, data_home=data_home[0], valid=True, audio_len=audio_len)
    else:
        val_set = DNSDataset(val_json_file, data_home=data_home, valid=True, audio_len=audio_len)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )
    return train_loader, val_loader
