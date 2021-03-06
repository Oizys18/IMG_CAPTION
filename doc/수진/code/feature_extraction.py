# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt


from ai_sub2.config import config


import numpy as np
import os
from tqdm import tqdm



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config_gpu = ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.allow_growth = True
config_gpu.gpu_options.allow_growth = True
session = InteractiveSession(config=config_gpu)
# 지금까지 생성된 tensor 그래프를 제거
tf.compat.v1.reset_default_graph()
tf.autograph.experimental.do_not_convert()

def get_data_file():
    data = np.load(os.path.join(config.base_dir, 'datasets', 'test_datasets.npy'))
    # test 50개만 
    # img_paths = data[:config.do_sampling, :1]
    # captions = data[:config.do_sampling, 2:]
    img_paths = data[:50, :1]
    captions = data[:50, 2:]
    train_images = np.squeeze(img_paths, axis=1)
    train_images = [os.path.join(config.base_dir, 'datasets', 'images', f'{img}') for img in train_images]
    train_captions = np.squeeze(captions, axis=1)
    train_captions = ['<start>' + cap + ' <end>' for cap in train_captions]
    return train_images, train_captions


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# file 불러오기
img_name_vector, train_captions = get_data_file()

# Inception V3 모델링
image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# Get unique images
encode_train = sorted(set(img_name_vector))


image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

for img, path in tqdm(image_dataset):
    batch_features = image_features_extract_model(img)

    batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(os.path.join(config.base_dir, 'datasets','features',os.path.basename(path_of_feature).replace('jpg', 'npy')), bf.numpy())