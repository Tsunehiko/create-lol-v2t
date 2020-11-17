NAME='resnext_tuned'
IN="../data/"
OUT="./frame"
LABEL="../label"
LOG="./results/resnext_tuned/"
WORKERS=12
BATCH=64
PYTHON="python"
SEED=31
EPOCHS=200
INTERVAL=10
MODEL="resnext"
THRESHOLD=0.8906274649611472
LR=0.002269239833144935

CUDA_VISIBLE_DEVICES=0 ${PYTHON} train_2d.py --video-path ${IN} --frame-path ${OUT} \
--label-path ${LABEL} --log ${LOG} --workers ${WORKERS} --batch ${BATCH} \
--seed ${SEED} --epochs ${EPOCHS} \
--model-name ${MODEL} --interval ${INTERVAL} \
--threshold ${THRESHOLD} --name ${NAME} --lr ${LR}