import tensorflow as tf
from config import config
from data import preprocess
import numpy as np
import os
from tqdm import tqdm
from .img_augmentation import img_aug
from pathlib import Path
from PIL import Image


def feature_extraction(base_dir, img_name_vector, train_captions, conf_img_aug):
    # 텐서 초기화
    tf.autograph.experimental.do_not_convert()
    tf.compat.v1.reset_default_graph()

    # 데이터 증강
    aug_img_name_vector = []
    aug_img_captions = []
    if conf_img_aug != 'original':
        for idx, img_name in enumerate(img_name_vector):
            augmented_img = img_aug(img_name, conf_img_aug)
            agimage = Image.fromarray(augmented_img)
            agimage.save(Path(base_dir, 'augmented_images', f'aug_{img_name}'))
            aug_img_name_vector += [os.path.join(base_dir, 'augmented_images', f'aug_{img_name}')]
            aug_img_captions += [train_captions[idx]]

    original_img_name_vector = [os.path.join(base_dir, 'images', f'{img}') for img in img_name_vector]
    img_name_vector = original_img_name_vector + aug_img_name_vector
    train_captions = train_captions + aug_img_captions

    # Inception V3 모델링
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    encode_train = list(set(img_name_vector))

    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
        preprocess.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)

        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            tmp_path = os.path.join(base_dir, 'features', os.path.basename(path_of_feature).replace('jpg', 'npy'))
            np.save(tmp_path, bf.numpy())

    return img_name_vector, train_captions