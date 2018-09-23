import joblib
from sys import argv
from os.path import exists
from scipy.stats import spearmanr as spr
from sklearn.metrics import mean_squared_error as mse
from models import model2
import os
import tensorflow as tf
import glob
import numpy as np
import pandas as pd

import os

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def prepare_image(_image_src, target_size):
    '''
        Takes image source as input as return
        processed image array ready for train/test/val
    :param _image_src: source of image
    :return: image_array
    '''
    img = image.load_img(_image_src, target_size = target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_session(gpu_fraction=0.6):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


if __name__ == '__main__':

    imgSrc = 'C:/dev/thorn/Aesthetic_attributes_maps-master/minitest/*'  # path to directory of images
    target_size = (299, 299)
    inputImages = glob.glob(imgSrc)
    n = len(inputImages)
    imageData = np.zeros((n, target_size[0], target_size[1], 3))

    for i, _image in enumerate(inputImages):
        x = prepare_image(_image, target_size)
        imageData[i, :, :, :] = x
        # if (i + 1) % 100 == 0:
        #     print (i, _image, x.shape)

    print(imageData.shape)
    batch_size = 6
    weights_file = "./weights-improvement__016-0.022715.hdf5"
    model = model2(weights_path=weights_file)
    _predict = model.predict(imageData, batch_size=batch_size, verbose=1)