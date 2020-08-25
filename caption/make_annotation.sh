PYTHON="python"
VIDEO_DIR="./temp/videos"
CAPTION_DIR="./temp/captions"
DIV_VIDEO_DIR="./temp/divided_videos"
ANNOTATION_DIR="./temp/annotation"
FRAME_DIR="./temp/frames"
PYSCENEDETECT_THRESHOLD=20
LOG="temp/log"

${PYTHON} make_annotation.py --video-dir ${VIDEO_DIR} --caption-dir ${CAPTION_DIR} \
--divided-video-dir ${DIV_VIDEO_DIR} --annotation-dir ${ANNOTATION_DIR} \
--frame-dir ${FRAME_DIR} --pyscenedetect-threshold ${PYSCENEDETECT_THRESHOLD} \
--log ${LOG}