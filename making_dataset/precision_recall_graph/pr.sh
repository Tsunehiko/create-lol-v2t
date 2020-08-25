NAME='resnext_pr'
IN="../data/"
OUT="../2D/frame"
LABEL="../label"
LOG="./resnext_pr/"
SAVE="./best.pkl"
WORKERS=12
PYTHON="python"
SEED=31
MODEL="resnext"

CUDA_VISIBLE_DEVICES=9 ${PYTHON} pr.py --video-path ${IN} --frame-path ${OUT} \
--label-path ${LABEL} --log ${LOG} --workers ${WORKERS} \
--seed ${SEED} --name ${NAME} --model-path ${SAVE} --model-name ${MODEL}e