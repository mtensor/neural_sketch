#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:45:52 2018

@author: Maxwell
"""

from scipy import misc
import skimage.transform as im
import matplotlib.pyplot as plt


def resize_img(image_in, output_pixel_size, crop_ratio):
    shape_in = image_in.shape
    assert shape_in[0] == shape_in[1]
    shape_in = shape_in[0]
    shape_out = int(shape_in * crop_ratio)
    


    start_loc = int((shape_in - shape_out) / 2.)
    image_cropped = image_in[start_loc:start_loc+shape_out, start_loc:start_loc+shape_out, :]
    image_out = im.resize(image_cropped,(output_pixel_size,output_pixel_size))
    return image_out

img = misc.imread('/Users/Maxwell/Downloads/aligned_detect_3.1047.jpg')
plt.imshow(img)

cropped_img = resize_img(img,40,.5)

plt.imshow(cropped_img)


