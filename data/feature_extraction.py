# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
from config import config
from data import preprocess
import numpy as np
import os
from tqdm import tqdm
from .img_augmentation import img_aug
from pathlib import Path
from PIL import Image


def feature_extraction():
    # 텐서 초기화
    tf.autograph.experimental.do_not_convert()
    tf.compat.v1.reset_default_graph()
    # file 불러오기
    img_name_vector, train_captions = preprocess.get_data_file()

    # 데이터 증강
    aug_img_name_vector = []
    if config.img_aug != 'original':
        for img_name in img_name_vector:
            augmented_img = img_aug(img_name, config.img_aug)

            # numpy로 저장
            # np.save(Path(config.base_dir,'datasets','augmented_images',f'{img_name}'),augmented_img)

            # jpg로 저장
            agimage = Image.fromarray(augmented_img)
            agimage.save(Path(config.base_dir, 'datasets',
                              'augmented_images', f'aug_{img_name}'))

            aug_img_name_vector += [os.path.join(
                config.base_dir, 'datasets', 'augmented_images', f'aug_{img_name}')]

    original_img_name_vector = [os.path.join(
        config.base_dir, 'datasets', 'images', f'{img}') for img in img_name_vector]
    img_name_vector = original_img_name_vector + aug_img_name_vector

    # Inception V3 모델링
    image_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # Get unique images
    encode_train = sorted(set(img_name_vector))

    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
        preprocess.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)

        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(os.path.join(config.base_dir, 'datasets', 'features', os.path.basename(
                path_of_feature).replace('jpg', 'npy')), bf.numpy())
