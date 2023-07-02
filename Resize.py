from PIL import Image
import numpy as np
import os, sys
import cv2

size = (224,224)

path = 'Data/Images/'
dirs = os.listdir( path )

for ii, item in enumerate(dirs):
    #image = cv2.imread(path + item, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(Image.open(path + item).convert("L"))
    print(f'Image {ii+1} shape: {image.shape}')
    imResize = cv2.resize(image, size)
    cv2.imwrite('Data/Images_resized/' + item, imResize)
    print(f'Image {ii+1} Done!')
