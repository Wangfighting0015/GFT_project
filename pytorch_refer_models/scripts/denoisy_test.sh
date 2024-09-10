PROJECT_HOME=/home/v-tianrwang/datas/tsstd01wus2_models/users/v-tianrwang/codes/XTeam/pytorch_refer_models
# 这个目录下需要有两个json和clean、noisy两个文件夹
NOISYWAV=/home/v-tianrwang/datas/tsstd01wus2_models/users/v-tianrwang/codes/XTeam/test_datas/noisy/book_04757_chp_0044_reader_08858_0_ZyQAy8PzgZ8_snr11_fileid_4.wav
SAVEPATH=/home/v-tianrwang/datas/tsstd01wus2_models/users/v-tianrwang/codes/XTeam/test_datas/est/book_04757_chp_0044_reader_08858_0_ZyQAy8PzgZ8_snr11_fileid_4.wav
MODEL_NAME=nsnet
CUDAIDX="0"
CHKPTPATH=/home/v-tianrwang/datas/tsstd01wus2_models/users/v-tianrwang/codes/XTeam/pytorch_refer_models/exp/nsnet_sisnr/_ckpt_epoch_81.ckpt
cd $PROJECT_HOME

# ---------------------------
# 非lava训练不用加这些
# CUDA_HOME=/usr/local/cuda-10.2 \
# PATH="$PATH:/usr/local/cuda-10.2/bin" \
# LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64" \
# PYTHONPATH=$PROJECT_HOME \
# ----------------------------

CUDA_VISIBLE_DEVICES=$CUDAIDX \
python -u denoisy_test.py \
--noisy-wav $NOISYWAV \
--model-name $MODEL_NAME \
--save-path $SAVEPATH \
--chptpath $CHKPTPATH