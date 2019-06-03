#!/usr/bin/env python
import torch
import cv2
import time
from core.detectors import CornerNet_Saccade
from core.detectors import CornerNet_Squeeze
from core.detectors import CornerNet
from core.vis_utils import draw_bboxes



# detector = CornerNet()
# detector1 = CornerNet_Saccade()
detector2 = CornerNet_Squeeze()

####

####
start = time.time()
image    = cv2.imread("moto.jpg")

# bboxes = detector(image)
# bboxes1 = detector1(image)
bboxes2 = detector2(image)

end = time.time()
print("time:", end-start)

# image  = draw_bboxes(image, bboxes)
# image1  = draw_bboxes(image, bboxes1)
image2  = draw_bboxes(image, bboxes2)
# cv2.imwrite("demo_out_corner.jpg", image)
# cv2.imwrite("demo_out_saccade.jpg", image1)
cv2.imwrite("demo_out_squeeze.jpg", image2)
