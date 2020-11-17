PYTHON="python"
EXP_NAME="interpolation_deepsegment"
VIDEO_DIR="../data/worlds_short/videos"
CAPTION_DIR="../data/worlds_short/captions"
TMP_DIR="./tmp"
DATASET_DIR="."
PYSCENEDETECT_THRESHOLD=20
LOG="./log/"
THREADS=28
WIDTH=340
HEIGHT=256
TASK='both'
METHOD="tvl1"
INTERVAL=16
CLASSIFY=/home/Tanaka/generate-commentary/caption/model/best_cpu.pkl
MODE='interpolation'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 ${PYTHON} make_dataset.py --exp-name ${EXP_NAME} --video-dir ${VIDEO_DIR} \
--caption-dir ${CAPTION_DIR} --tmp-dir ${TMP_DIR} \
--dataset-dir ${DATASET_DIR} \
--log ${LOG} --pyscenedetect-threshold ${PYSCENEDETECT_THRESHOLD} \
--threads ${THREADS} --flow-type ${METHOD} --task ${TASK} --frame-interval ${INTERVAL} \
--classify-model ${CLASSIFY} --mode ${MODE} \
--new-width ${WIDTH} --new-height ${HEIGHT}