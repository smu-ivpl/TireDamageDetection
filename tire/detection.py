# disable warning
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import cv2
import skimage
import datetime
import matplotlib.pyplot as plt
import argparse
import torch
import torch.backends.cudnn as cudnn
import os
import glob
import numpy as np
import string

# mrcnn
from tire.mrcnn.config import Config
from tire.mrcnn import utils, model as modellib, visualize
# tire
from tire import detect_dot, detect_wear, detect_defect, detect_text

# -- crop roi
def crop_bbox(image, bbox, type):
    rois = []

    if type == 'tread':
        pvalue = 0
    elif type == 'sidewall':
        pvalue = 20

    for b in bbox:
        # padding
        p = [b[0]-pvalue,b[2]+pvalue, b[1]-pvalue,b[3]+pvalue]
        if p[0] < 0: p[0] = 0
        if p[2] < 0: p[2] = 0
        if p[1] > image.shape[0]: p[1] = image.shape[0]
        if p[3] > image.shape[1]: p[3] = image.shape[1]

        roi = image[p[0]:p[1], p[2]:p[3]]
        rois.append(roi)

        # visualize
        #plt.imshow(roi)
        #plt.show()

    return rois


# -- tread and sidwall detection
class DetectionConfig(Config):
    NAME = "tire"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    DETECTION_MIN_CONFIDENCE = 0.99

def detect_tire(image, type):
    config = DetectionConfig()

    # load img
    image = image

    # load model
    if type == 'tread':
        weights_path = "tire/model/tread_model.h5"
    elif type == 'sidewall':
        weights_path = "tire/model/sidewall_model.h5"

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="logs")
    model.load_weights(weights_path, by_name=True)
    r = model.detect([image], verbose=0)[0]


    # bounding box
    class_names = ['background', type ]
    bbox = utils.extract_bboxes(r['masks'])

    # check detection
    N = bbox.shape[0]
    if not N:
        print("*** "+type+" 없음 *** ")
        return None, False
    else:
        # visualize.display_instances(image, bbox, r['masks'], r['class_ids'], class_names, r['scores'])
        rois = crop_bbox(image, bbox, type)
        return rois[0], True



# -- tread input
def tread(img):

    # -- 1. 손상 부위 검출
    crop_tread, bool = detect_tire(img, type='tread')

    if bool :   # true
        print("\n*** 트레드 검출 ***")
        detect_defect.defectModel(crop_tread)

        # write log

        # -- 2. wear
        detect_wear.wearModel(crop_tread)

        # write log
    else:
        print("\n*** 트레드 없음 ***\n")



# -- 사이드월 input
def sidewall(img):

    # -- 1. defects detection
    crop_sidewall, bool = detect_tire(img, type='sidewall')
    if bool :   # true
        print("\n*** 사이드월 검출 ***")
        detect_defect.defectModel(crop_sidewall)

        # -- 2. tire  모델 인식
        # -- 2.1. DOT 영역 검출
        # histo
        grayimg = cv2.cvtColor(crop_sidewall, cv2.COLOR_BGR2GRAY)
        histo = cv2.equalizeHist(grayimg)
        histo = cv2.cvtColor(histo, cv2.COLOR_GRAY2BGR)

        #plt.imshow(histo)
        #plt.show()

        crop_dots, bool2 = detect_dot.dotModel(histo, crop_sidewall)


        if bool2:   # DOT true

            files = glob.glob('temp/*')
            for fn in files:
                os.remove(fn)

            # save dot
            for i, dot_img in enumerate(crop_dots):
                #plt.imshow(i)
                #plt.show()
                # save_path = "tire/result/" + "dot_{:%Y%m%dT%H%M%S}.jpg".format(datetime.datetime.now())
                skimage.io.imsave("temp/%d.jpg" % i, np.uint8(dot_img * 255))
            
            # -- 2.2. DOT Text Recognition
            detect_text.demo()

    else:
        print("\n*** 사이드월 없음 ***\n")



