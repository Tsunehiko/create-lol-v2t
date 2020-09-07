PYTHON="python"
VIDEO_DIR="./tmp/videos"
CAPTION_DIR="./tmp/captions"
DIV_VIDEO_DIR="./tmp/divided_videos"
ANNOTATION_DIR="./tmp/annotation"
FRAME_DIR="./tmp/frames"
PYSCENEDETECT_THRESHOLD=20
LOG="tmp/log"

${PYTHON} make_annotation.py --video-dir ${VIDEO_DIR} --caption-dir ${CAPTION_DIR} \
--divided-video-dir ${DIV_VIDEO_DIR} --annotation-dir ${ANNOTATION_DIR} \
--frame-dir ${FRAME_DIR} --pyscenedetect-threshold ${PYSCENEDETECT_THRESHOLD} \
--log ${LOG}