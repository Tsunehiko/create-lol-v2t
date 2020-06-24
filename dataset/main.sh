IN="./data"
OUT="./frame"
LABEL="./label"
LOG="./log"
WORKERS=12
BATCH=16
PYTHON="python"
CLIP=16
SEED=31
EPOCHS=200
INTERVAL=10

CUDA_VISIBLE_DEVICES=6,7,8,9 ${PYTHON} train.py --video-path ${IN} --frame-path ${OUT} \
--label-path ${LABEL} --batch ${BATCH} --workers ${WORKERS} \
--seed ${SEED} --clip-len ${CLIP} --epochs ${EPOCHS} --log ${LOG} \
--interval ${INTERVAL}
