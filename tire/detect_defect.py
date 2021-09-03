''' detect tire defects using mask rcnn model'''

import datetime
import skimage

# Import Mask RCNN
from tire.mrcnn.config import Config
from tire.mrcnn import utils, model as modellib, visualize


class InferenceConfig(Config):
    NAME = "tire"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    DETECTION_MIN_CONFIDENCE = 0.99


def defectModel(image):

    config = InferenceConfig()
    #config.display()

    # load img
    image=image

    # load model
    weights_path = "tire/model/defect_model.h5"
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="logs")
    model.load_weights(weights_path, by_name=True)
    r = model.detect([image], verbose=0)[0]

    # bounding box
    class_names = ['background', 'defect']
    bbox = utils.extract_bboxes(r['masks'])

    N = bbox.shape[0]
    if not N:
        print("*** 손상 부위 없음 *** ")
        return

    else:
        # visualize
        output = visualize.display_instances(image, bbox, r['masks'], r['class_ids'], class_names, r['scores'])

        # save
        save_path = "tire/result/" + "defect_{:%Y%m%dT%H%M%S}.jpg".format(datetime.datetime.now())
        skimage.io.imsave(save_path, output)

        print("*** 손상 부위 검출 이미지 저장 경로 : "+save_path+"   *** ")


def predict(image):
    config = InferenceConfig()
    image = image

    # load model
    weights_path = "tire/model/defect_model.h5"
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="logs")
    model.load_weights(weights_path, by_name=True)
    r = model.detect([image], verbose=0)[0]

    # bounding box
    class_names = ['background', 'defect']
    bbox = utils.extract_bboxes(r['masks'])

    N = bbox.shape[0]
    if not N:
        return None

    else:
        import matplotlib.pyplot as plt
        return visualize.display_instances(image,
                                           bbox,
                                           r['masks'],
                                           r['class_ids'],
                                           class_names,
                                           r['scores'],
                                           ax=plt.subplots(1, figsize=(16,16))[1])