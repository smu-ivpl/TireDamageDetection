''' detect tire defects using mask rcnn model'''

import skimage
import matplotlib.pyplot as plt
import cv2

# Import Mask RCNN
from tire.mrcnn.config import Config
from tire.mrcnn import utils, model as modellib, visualize


class InferenceConfig(Config):
    NAME = "dot"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    DETECTION_MIN_CONFIDENCE = 0.9


# crop dot rois
def crop_bbox(image, bbox):
    rois = []

    for b in bbox:
        # padding
        p = [b[0] - 10, b[2] + 10, b[1] - 10, b[3] + 10]
        if p[0] < 0: p[0] = 0
        if p[2] < 0: p[2] = 0
        if p[1] > image.shape[0]: p[1] = image.shape[0]
        if p[3] > image.shape[1]: p[3] = image.shape[1]

        # skimage crop -> image[x1:x2,y1:y2]
        roi = image[p[0]:p[1], p[2]:p[3]]

        # cordinate to rotate dot images
        height, width, depth = image.shape
        mid_x = (b[1] + b[3]) / 2
        mid_y = (b[0] + b[2]) / 2

        # calculate rotate based dot location on tire image
        # -- left
        if mid_x < width / 3:
            # up
            if mid_y < height / 3:
                rotate_roi = skimage.transform.rotate(roi, -45, resize=True)
                rois.append(rotate_roi)

            # middle
            elif mid_y < 2 * height / 3:
                roi = skimage.transform.rotate(roi, -90, resize=True)
                rois.append(roi)

            # down
            else:
                rotate_roi = skimage.transform.rotate(roi, -135, resize=True)
                rois.append(rotate_roi)

        # -- right
        elif mid_x > 2 * width / 3:
            # up
            if mid_y < height / 3:
                rotate_roi = skimage.transform.rotate(roi, 45, resize=True)
                rois.append(rotate_roi)
            # middle
            elif mid_y < 2 * height / 3:
                roi = skimage.transform.rotate(roi, 90, resize=True)
                rois.append(roi)
            # down
            else:
                rotate_roi = skimage.transform.rotate(roi, 135, resize=True)
                rois.append(rotate_roi)

        # -- middle
        else:
            # down
            if mid_y > height / 2:
                roi = skimage.transform.rotate(roi, 180, resize=True)
                rois.append(roi)
            else:
                rois.append(roi)

    return rois


def dotModel(histo, crop_sidewall):

    config = InferenceConfig()
    #config.display()

    # load img
    histo=histo

    # load model
    weights_path = "tire/model/histo_dot_model.h5"
    model = modellib.MaskRCNN(mode="inference", config=config,model_dir="logs")
    model.load_weights(weights_path, by_name=True)
    r = model.detect([histo], verbose=0)[0]

    # bounding box
    class_names = ['background', 'dot']
    bbox = utils.extract_bboxes(r['masks'])

    # 존재 여부
    N = bbox.shape[0]
    if not N:
        print("\n*** DOT 없음 *** \n")
        return None, False

    else:
        visualize
        visualize.display_instances(crop_sidewall, bbox, r['masks'], r['class_ids'], class_names, r['scores'])
        rois = crop_bbox(crop_sidewall, bbox)

        print("*** DOT 검출 완료 ***")

        return rois, True



def predict(image):
    # histo
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histo = cv2.equalizeHist(grayimg)
    histo = cv2.cvtColor(histo, cv2.COLOR_GRAY2BGR)

    config = InferenceConfig()

    # load model
    weights_path = "tire/model/histo_dot_model.h5"
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="logs")
    model.load_weights(weights_path, by_name=True)
    r = model.detect([histo], verbose=0)[0]

    # bounding box
    class_names = ['background', 'dot']
    bbox = utils.extract_bboxes(r['masks'])

    # 존재 여부
    N = bbox.shape[0]
    if not N:
        return None

    else:
        dot_image = visualize.display_instances(image,
                                                bbox,
                                                r['masks'],
                                                r['class_ids'],
                                                class_names,
                                                r['scores'],
                                                ax=plt.subplots(1, figsize=(16, 16))[1])

        rois = crop_bbox(image, bbox)

        return dot_image, rois