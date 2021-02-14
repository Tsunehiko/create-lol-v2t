PYTHON="python"
VIDEO_DIR="../data/worlds_large/videos"
CAPTION_DIR="../data/worlds_large/captions"
DIV_VIDEO_DIR="../dataset/tmp/large/divide/training"
ANNOTATION_DIR="../dataset/large/annotation/deepsegment"
FRAME_DIR="../dataset/tmp/large/frame/training"
TIMECODE_DIR="../dataset/tmp/large/timecode/training"
TRASH_DIR="../dataset/tmp/large/trash/training"
TMP_ANNOTATION_DIR="../dataset/tmp/large/annotation/training"
PYSCENEDETECT_THRESHOLD=20
LOG="../dataset/log/large"
THREADS=15
PUNCT='deepsegment'
CLASSIFY=/home/Tanaka/generate-commentary/caption/model/best_cpu.pkl
MODE='interpolation'

${PYTHON} make_annotation_parallel.py --video-dir ${VIDEO_DIR} --caption-dir ${CAPTION_DIR} \
--divided-video-dir ${DIV_VIDEO_DIR} --annotation-dir ${ANNOTATION_DIR} --trash-dir ${TRASH_DIR} \
--frame-dir ${FRAME_DIR} --timecode-dir ${TIMECODE_DIR} --tmp-annotation-dir ${TMP_ANNOTATION_DIR} \
--pyscenedetect-threshold ${PYSCENEDETECT_THRESHOLD} --log ${LOG} \
--threads ${THREADS} --classify-model ${CLASSIFY} --mode ${MODE} --punct ${PUNCT}