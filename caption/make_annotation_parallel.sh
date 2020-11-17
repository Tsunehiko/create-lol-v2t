PYTHON="python"
VIDEO_DIR="../data/test/videos"
CAPTION_DIR="../data/test/captions"
DIV_VIDEO_DIR="./tmp/test_interpolation/divided_videos"
ANNOTATION_DIR="./tmp/test_interpolation/annotation"
FRAME_DIR="./tmp/test_interpolation/frame"
TIMECODE_DIR="./tmp/test_interpolation/timecode"
PYSCENEDETECT_THRESHOLD=20
LOG="./tmp/test_interpolation/log"
THREADS=48
PUNCT='deepsegment'
CLASSIFY=/home/Tanaka/generate-commentary/caption/model/best_cpu.pkl
MODE='interpolation'

${PYTHON} make_annotation_parallel.py --video-dir ${VIDEO_DIR} --caption-dir ${CAPTION_DIR} \
--divided-video-dir ${DIV_VIDEO_DIR} --annotation-dir ${ANNOTATION_DIR} \
--frame-dir ${FRAME_DIR} --timecode-dir ${TIMECODE_DIR} \
--pyscenedetect-threshold ${PYSCENEDETECT_THRESHOLD} --log ${LOG} \
--threads ${THREADS} --classify-model ${CLASSIFY} --mode ${MODE} --punct ${PUNCT}