PROJECT_HOME=/home/v-tianrwang/datas/tsstd01wus2_models/users/v-tianrwang/codes/XTeam/pytorch_refer_models
# 这个目录下需要有两个json和clean、noisy两个文件夹
DNSHOME=/home/v-tianrwang/datas/tsstd01wus2_models/users/v-tianrwang/codes/XTeam/no_reverb
# CSVSAVEPATH=/home/v-tianrwang/datas/tsstd01wus2_models/users/v-tianrwang/codes/XTeam/pytorch_refer_models/exp/nsnet_sisnr/dns.csv
CSVSAVEPATH=/home/v-tianrwang/datas/tsstd01wus2_models/users/v-tianrwang/codes/XTeam/pytorch_refer_models/exp/dpcrn_sisnr/dns.csv
# MODEL_NAME=nsnet
MODEL_NAME=dpcrn
CUDAIDX="0"
# CHKPTPATH=/home/v-tianrwang/datas/tsstd01wus2_models/users/v-tianrwang/codes/XTeam/pytorch_refer_models/exp/nsnet_sisnr/_ckpt_epoch_81.ckpt
CHKPTPATH=/home/v-tianrwang/datas/tsstd01wus2_models/users/v-tianrwang/codes/XTeam/pytorch_refer_models/exp/dpcrn_sisnr/_ckpt_epoch_110.ckpt
cd $PROJECT_HOME

# ---------------------------
# 非lava训练不用加这些
# CUDA_HOME=/usr/local/cuda-10.2 \
# PATH="$PATH:/usr/local/cuda-10.2/bin" \
# LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64" \
# PYTHONPATH=$PROJECT_HOME \
# ----------------------------

CUDA_VISIBLE_DEVICES=$CUDAIDX \
python -u eval_on_dns.py \
--dns-test $DNSHOME \
--model-name $MODEL_NAME \
--csv-save-path $CSVSAVEPATH \
--chptpath $CHKPTPATH