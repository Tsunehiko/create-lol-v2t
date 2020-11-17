PYTHON="python"
VIDEO_DIR="./tmp/videos"
CAPTION_DIR="./tmp/captions"
DIV_VIDEO_DIR="./tmp/divided_videos/series"
ANNOTATION_DIR="./tmp/annotation/series"
FRAME_DIR="./tmp/frames"
TIMECODE_DIR="tmp/timecodes/series"
PYSCENEDETECT_THRESHOLD=20
LOG="tmp/log/series"

${PYTHON} make_annotation.py --video-dir ${VIDEO_DIR} --caption-dir ${CAPTION_DIR} \
--divided-video-dir ${DIV_VIDEO_DIR} --annotation-dir ${ANNOTATION_DIR} \
--frame-dir ${FRAME_DIR} --timecode-dir ${TIMECODE_DIR} \
--pyscenedetect-threshold ${PYSCENEDETECT_THRESHOLD} --log ${LOG}