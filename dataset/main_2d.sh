IN="./data/"
OUT="./2D/frame"
LABEL="./label"
LOG="./2D/resnet_2/log"
WORKERS=12
BATCH=16
PYTHON="python"
SEED=31
EPOCHS=200
INTERVAL=10
MODEL="resnet"

CUDA_VISIBLE_DEVICES=0 ${PYTHON} train_2d.py --video-path ${IN} --frame-path ${OUT} \
--label-path ${LABEL} --log ${LOG} --workers ${WORKERS} --batch ${BATCH} \
--seed ${SEED} --epochs ${EPOCHS} \
--model-name ${MODEL} --interval ${INTERVAL}