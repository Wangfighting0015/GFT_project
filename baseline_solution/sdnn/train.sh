# CUDA_VISIBLE_DEVICES=1 python -u train.py \
# --data-home /CDShare3/2023/wangtianrui/DNS/5s_100h \
# --save-home exp \
# --batch-size 32 \
# --valid-num 1 \
# --audio-len 80000

mkdir -p ./exp

CUDA_HOME=/usr/local/cuda-10.2 \
PATH="$PATH:/usr/local/cuda-10.2/bin" \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64" \
PYTHONPATH=/home/wang/codes/py/XTeam/baseline_solution/sdnn \
CUDA_VISIBLE_DEVICES=1 \
python -u train.py \
--data-home /home/wang/datasets/DNS-Challenge/DNS-Challenge/make_data/ \
--save-home exp \
--batch-size 42 \
--valid-num 1 \
--audio-len 480000  2>&1 | tee ./exp/log.out

# data-home : ../../test_data/ 