# import joblib
# from sys import argv
# from os.path import exists
# from scipy.stats import spearmanr as spr
# from sklearn.metrics import mean_squared_error as mse
from models import model2
import os
import tensorflow as tf
import glob
import numpy as np
import pandas as pd
from PIL import Image
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

    # imgSrc = 'C:\dev/thorn\spotlight-sample-data\image_data_sample/*'  # path to directory of images
    # target_size = (299, 299)
    # inputImages = glob.glob(imgSrc)
    # #inputImages = ALL_inputImages[1:100]
    # n = len(inputImages)
    # imageData = np.zeros((n, target_size[0], target_size[1], 3))
    #
    # for i, _image in enumerate(inputImages):
    #     x = prepare_image(_image, target_size)
    #     imageData[i, :, :, :] = x
    #     # if (i + 1) % 100 == 0:
    #     #     print (i, _image, x.shape)
    #
    # print(imageData.shape)
    # batch_size = 100
    # weights_file = "./weights-improvement__016-0.022715.hdf5"
    # model = model2(weights_path=weights_file)
    # _predict = model.predict(imageData, batch_size=batch_size, verbose=1)
    #
    # print(_predict[-1])


   # new_data = pd.DataFrame({'filename':inputImages, 'score':_predict[-1].tolist()})
    new_data = pd.read_csv("ALL__predict.csv")
 #   print(new_data)
    #new_data.to_csv('ALL__predict.csv', index=False)

    # ims_per_group = 20
    #
    # Ordered_df =
    # Most_Pro_df =
    # Least_Pro_df =
    #
    # Most_Pro_Images = []
    # Least_Pro_Images = []


    ims_per_group = 20
    x= 750
    ordered = new_data.sort_values('score', 0, False)
    pro_names = ordered['filename'][x:(x+ims_per_group)].tolist()
    #non_pro_names = ordered['filename'][-1*(ims_per_group +1):-1].tolist()
    # mini_ordered = new_data.sort_values('score', 0, False)
    # names = mini_ordered['filename'][0:3].tolist()


    rows = ims_per_group // 5 + 1
    pro_ims = Image.new('RGB', (200*5, 200*rows))

    x_offset = 0
    y_offset = 0
    counter = 0

    for image in pro_names:
        img = Image.open(image, "r")
        img = img.resize([200, 200])
        if counter <= 4:
            pro_ims.paste(img, (x_offset, y_offset))
            x_offset = x_offset + 200
            counter = counter + 1
        else:
            x_offset = 0
            y_offset = y_offset + 300
            pro_ims.paste(img, (x_offset, y_offset))
            x_offset = x_offset + 200
            counter = 0

    pro_ims.show()
    #
    #
    #
    # non_pro_ims =  Image.new('RGB', (200*5, 200*rows))
    #
    # for image in Least_Pro_Images:
    #     img = Image.open(image, "r")
    #     img = img.resize([200, 200])
    #     if counter <= 4:
    #         non_pro_ims.paste(img, (x_offset, y_offset))
    #         x_offset = x_offset + 200
    #         counter = counter + 1
    #     else:
    #         x_offset = 0
    #         y_offset = y_offset + 300
    #         non_pro_ims.paste(img, (x_offset, y_offset))
    #         x_offset = x_offset + 200
    #         counter = 0
    #
    # non_pro_ims.show()




    # for image in pro_names:
    #     img = Image.open(image, "r")
    #     img = img.resize([200, 200])
    #     if counter <= 4:
    #         pro_ims.paste(img, (x_offset, y_offset))
    #         x_offset = x_offset + 200
    #         counter = counter + 1
    #     else:
    #         x_offset = 0
    #         y_offset = y_offset + 300
    #         pro_ims.paste(img, (x_offset, y_offset))
    #         x_offset = x_offset + 200
    #         counter = 0
    #
    # pro_ims.show()

    # for image in inputImages:
    #     img = Image.open(image, "r")
    #     img = img.resize([200, 200])
    #     if counter <= 4:
    #         pro_ims.paste(img, (x_offset, y_offset))
    #         x_offset = x_offset + 200
    #         counter = counter +1
    #     else:
    #         x_offset = 0
    #         y_offset = y_offset + 300
    #         pro_ims.paste(img, (x_offset, y_offset))
    #         x_offset = x_offset + 200
    #         counter = 0
    #
    #
    # pro_ims.show()
    # new_im.save('full_image.png')

