# 테스트

import skimage
import matplotlib.pyplot as plt

# tire module
from tire import detection      # input tread image

# load image
tread_image = skimage.io.imread('/home/user/TireProject/testImage/tread.jpg')
tread_left_image = skimage.io.imread('/home/user/TireProject/testImage/tread_1.6.jpg')
sidewall_image = skimage.io.imread('/home/user/TireProject/testImage/sidewall.jpg')

#tread_defect_image = skimage.io.imread('/home/user/TireProject/testImage/tread_defect3.jpg')


# defects detection model
detection.tread(tread_image)
detection.tread(tread_left_image)
detection.sidewall(sidewall_image)

#detection.tread(tread_defect_image)