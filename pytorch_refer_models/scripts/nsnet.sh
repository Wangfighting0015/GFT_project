PROJECT_HOME=/home/v-tianrwang/datas/tsstd01wus2_models/users/v-tianrwang/codes/XTeam/pytorch_refer_models
# 这个目录下需要有两个json和clean、noisy两个文件夹
DATA_HOME=/home/v-tianrwang/datas/dns/5s_100h
MODEL_NAME=nsnet
LOSS_NAME=sisnr
BATCHSIZE=1024
CUDAIDX="0"
LR=1e-3
LR_DECEND_FAC=0.75
STOP_PATIENCE=20
PRETRAINED_CKPT=-1   # 用来从LOG_HOME读取预训练好的checkpoint，输入对应的编号就行，负数则不读取

LOG_HOME=$PROJECT_HOME/exp/${MODEL_NAME}_${LOSS_NAME}
cd $PROJECT_HOME

mkdir -p $LOG_HOME

# ---------------------------
# 非lava训练不用加这些
# CUDA_HOME=/usr/local/cuda-10.2 \
# PATH="$PATH:/usr/local/cuda-10.2/bin" \
# LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64" \
# PYTHONPATH=$PROJECT_HOME \
# ----------------------------

CUDA_VISIBLE_DEVICES=$CUDAIDX \
python -u train.py \
--data-home $DATA_HOME \
--model-name $MODEL_NAME \
--loss-name $LOSS_NAME \
--save-home $LOG_HOME \
--batch-size $BATCHSIZE \
--load-param-index $PRETRAINED_CKPT \
--lr $LR \
--lr-descend-factor $LR_DECEND_FAC \
--stop-patience $STOP_PATIENCE  2>&1 | tee ${LOG_HOME}/train.log