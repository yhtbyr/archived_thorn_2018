import glob
import itertools
import numpy as np
import pandas as pd
from keras import applications
from collections import defaultdict
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def prepare_image(_image_src, target_size):
    img = image.load_img(_image_src, target_size = target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def compare(x,y):
    dist = np.sqrt(np.sum(np.square(np.subtract(x, y))))
    return (dist)

if __name__ == '__main__':

    #imgSrc = 'C:\dev/thorn\spotlight-sample-data\image_data_sample/*'
    imgSrc = 'C:/dev/thorn/Aesthetic_attributes_maps-master/minitest/*'
    target_size = (224, 224)
    inputImages = glob.glob(imgSrc)

    print(inputImages)
    splits = [x.split("\\")[1] for x in inputImages]
    #splits = [x.split("\\")[4] for x in inputImages] ## for full dataset

    imageData = np.zeros((len(inputImages), target_size[0], target_size[1], 3))
    for i, _image in enumerate(inputImages):
        x = prepare_image(_image, target_size)
        imageData[i, :, :, :] = x
    print(imageData.shape)

    #vector_data = pd.DataFrame({'filename': inputImages, 'vector': _predict.tolist()})


    batch_size = len(inputImages)
    model = applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling='max')
    _predict = model.predict(imageData, batch_size=batch_size, verbose=1)
    print(_predict.shape)

    ads = defaultdict(dict)
    for i, name in enumerate(inputImages):  ## x = full path
        full_image_name = name.split("\\")[1]
        #full_image_name = name.split("\\")[4]  ## for full dataset
        image_name = full_image_name.split("_")[0]
        image_number = (full_image_name.split("_")[1]).strip(".jpg")
        ads[image_name][image_number] = _predict[i]

    print(len(ads))

#import json
# with open('bigdict.txt', 'w') as file:
#     file.write(json.dumps(ads))

pd.DataFrame()

names = []
average_similarities = []
min_similarities = []
max_similarities = []
series_score = []

for ad in ads.keys():
    pics = ads[ad].keys()
    print(pics)
    pairs_of_pics = list(itertools.combinations(range(len(pics)), 2))
    print(pairs_of_pics)
    pic_scores = []
    names.append(ad)
    if len(pairs_of_pics) > 0:
        for p in pairs_of_pics:
            print(ads[ad])
            #print(ads[ad][0])
            #print("***************    " , ads[ad].keys()[p[0]])
            pic_scores.append(compare(ads[ad].keys()[p[0]], ads[ad].keys()[p[1]]))

        av = np.average(pic_scores)
        average_similarities.append(av)

        min = np.min(pic_scores)
        min_similarities.append(min)

        max = np.max(pic_scores)
        max_similarities.append(max)

        series_score_add = sum(pic_scores)
        series_score.append(series_score_add)

    else:
        average_similarities.append(0)
        min_similarities.append(0)
        max_similarities.append(0)
        series_score.append(0)

result = pd.DataFrame({'name': names, 'avg':average_similarities, 'min' :min_similarities, 'max': max_similarities, 'score' :series_score})
print(result)



## really you could compute the average similarity for unrelated images and have a set threshold for similarity








    # print(splits)
    # ads = {}
    # for x in splits:
    #     ad = x.split("\\")[2]
    #     if ad not in ads.keys():
    #         ads[ad] = []
    #     ads[ad] = x.split("\\")[3]


    # splits2 = [x.split("\\")[2] for x in splits]
    # unique_ads = list(set(splits2))
    # print(splits2)
    # print(len(splits2))
    # print(len(unique_ads))
    #
    #
        # if x.split("\\")[2] not in ads.keys():
        #     ads[x.split("\\")[2]] = {}
        #     ads[x.split("\\")[2]] = x.split("\\")[3]
        # else:
        #     ads[x.split("\\")[2]] =

    #new_data.to_csv('similarity_test_predict.csv', index=False)





