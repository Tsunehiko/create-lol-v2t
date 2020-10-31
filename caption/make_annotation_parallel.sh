PYTHON="python"
VIDEO_DIR="./tmp/test/videos"
CAPTION_DIR="./tmp/test/captions"
DIV_VIDEO_DIR="./tmp/test/divided_videos/parallel"
ANNOTATION_DIR="./tmp/test/annotation/parallel"
FRAME_DIR="./tmp/test/frames"
TIMECODE_DIR="tmp/test/timecodes/parallel"
PYSCENEDETECT_THRESHOLD=20
LOG="tmp/test/log/parallel"
THREADS=3

${PYTHON} make_annotation_parallel.py --video-dir ${VIDEO_DIR} --caption-dir ${CAPTION_DIR} \
--divided-video-dir ${DIV_VIDEO_DIR} --annotation-dir ${ANNOTATION_DIR} \
--frame-dir ${FRAME_DIR} --timecode-dir ${TIMECODE_DIR} \
--pyscenedetect-threshold ${PYSCENEDETECT_THRESHOLD} --log ${LOG} \
--threads ${THREADS}