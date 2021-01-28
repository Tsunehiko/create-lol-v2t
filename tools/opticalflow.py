import numpy as np
import cv2

frame0_path = "path_0"
frame1_path = "path_1"


def flow2img(flow, BGR=True):
    x, y = flow[..., 0], flow[..., 1]
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype = np.uint8)
    ma, an = cv2.cartToPolar(x, y, angleInDegrees=True)
    hsv[..., 0] = (an / 2).astype(np.uint8)
    hsv[..., 1] = (cv2.normalize(ma, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)).astype(np.uint8)
    hsv[..., 2] = 255
    if BGR:
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img


im1 = cv2.imread(frame0_path)
im2 = cv2.imread(frame1_path)
gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

dtvl1 = cv2.optflow.createOptFlow_DualTVL1()
flowDTVL1 = dtvl1.calc(gray1, gray2, None)

deepF = cv2.optflow.createOptFlow_DeepFlow()
flowDeep = deepF.calc(gray1, gray2, None)
cv2.imwrite("opticalflow.png", flow2img(flowDeep, False))
# cv2.optflow.writeOpticalFlow("dp.flo", flowDeep)
