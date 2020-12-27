PYTHON="python"
EXP_NAME="large"
VIDEO_DIR="../data/worlds_large/videos"
CAPTION_DIR="../data/worlds_large/captions"
TMP_DIR="./tmp"
DATASET_DIR="."
PYSCENEDETECT_THRESHOLD=15
LOG="./log/"
THREADS=52
WIDTH=340
HEIGHT=256
TASK='both'
METHOD="tvl1"
INTERVAL=13
CLASSIFY=/home/Tanaka/generate-commentary/caption/model/best_cpu.pkl
MODE='interpolation'
PUNCT='deepsegment'

CUDA_VISIBLE_DEVICES=0 ${PYTHON} make_dataset.py --exp-name ${EXP_NAME} --video-dir ${VIDEO_DIR} \
--caption-dir ${CAPTION_DIR} --tmp-dir ${TMP_DIR} \
--dataset-dir ${DATASET_DIR} \
--log ${LOG} --pyscenedetect-threshold ${PYSCENEDETECT_THRESHOLD} \
--threads ${THREADS} --flow-type ${METHOD} --task ${TASK} --frame-interval ${INTERVAL} \
--classify-model ${CLASSIFY} --mode ${MODE} --punct ${PUNCT} \
--new-width ${WIDTH} --new-height ${HEIGHT}