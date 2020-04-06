# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import os
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from tqdm import tqdm



def get_data_file():
    data = np.load('../../../datasets/test_datasets.npy')
    img_paths = data[:50, :1]
    captions = data[:50, 2:]
    train_images = np.squeeze(img_paths, axis=1)
    train_images = ['../../../datasets/images/' + img for img in train_images]
    train_captions = np.squeeze(captions, axis=1)
    train_captions = ['<start>' + cap + ' <end>' for cap in train_captions]
    return train_images, train_captions


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


img_name_vector, train_captions = get_data_file()

image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# Get unique images
encode_train = sorted(set(img_name_vector))



image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
# print(image_dataset)
# <TensorSliceDataset shapes: (), types: tf.string>
# print(image_dataset.enumerate())
# <DatasetV1Adapter shapes: ((), ()), types: (tf.int64, tf.string)>

image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

print(image_dataset)
# <BatchDataset shapes: ((None, 299, 299, 3), (None,)), types: (tf.float32, tf.string)>
for img, path in tqdm(image_dataset):
    batch_features = image_features_extract_model(img)

    batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))
    # print()
    # print('=======batch_features=====================')
    # print(batch_features)
    # print('for/for')
    for bf, p in zip(batch_features, path):
        # print()
        # print('p', p)
        path_of_feature = p.numpy().decode("utf-8")
        image_name = path_of_feature.split('/')[-1].split('.')[0]
        # print(path_of_feature)
        # print(bf.numpy())
        np.save(f'C:\\Users\\multicampus\\ai_sub2\\datasets\\features\\{image_name}.npy', bf.numpy())