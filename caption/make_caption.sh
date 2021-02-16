PYTHON="python"
VIDEO_DIR="../data/lol/videos"
CAPTION_DIR="../data/lol/captions"
SPLIT_VIDEO_DIR="./tmp/lol/split"
ANNOTATION_DIR="./tmp/lol/annotation"
FRAME_DIR="./tmp/lol/frame"
TIMECODE_DIR="./tmp/lol/timecode"
TRASH_DIR="./tmp/lol/trash"
TMP_ANNOTATION_DIR="./tmp/lol/annotation"
PYSCENEDETECT_THRESHOLD=20
LOG="./tmp/lol/log"
THREADS=50
PUNCT='deepsegment'
CLASSIFY=/home/Tanaka/generate-commentary/caption/model/best_cpu.pkl
MODE='interpolation'

${PYTHON} make_caption.py --video-dir ${VIDEO_DIR} --caption-dir ${CAPTION_DIR} \
--split-video-dir ${SPLIT_VIDEO_DIR} --annotation-dir ${ANNOTATION_DIR} --trash-dir ${TRASH_DIR} \
--frame-dir ${FRAME_DIR} --timecode-dir ${TIMECODE_DIR} --tmp-annotation-dir ${TMP_ANNOTATION_DIR} \
--pyscenedetect-threshold ${PYSCENEDETECT_THRESHOLD} --log ${LOG} \
--threads ${THREADS} --classify-model ${CLASSIFY} --mode ${MODE} --punct ${PUNCT}