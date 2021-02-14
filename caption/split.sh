PYTHON="python"
VIDEO_DIR="../data/worlds_short/videos"
DIV_VIDEO_DIR="../dataset/tmp/interpolation_deepsegment/divide"
TIMECODE_DIR="../dataset/tmp/interpolation_deepsegment/timecode"
PYSCENEDETECT_THRESHOLD=20
THREADS=48

${PYTHON} divide.py --video-dir ${VIDEO_DIR} \
--divided-video-dir ${DIV_VIDEO_DIR} --timecode-dir ${TIMECODE_DIR} \
--pyscenedetect-threshold ${PYSCENEDETECT_THRESHOLD} --threads ${THREADS}