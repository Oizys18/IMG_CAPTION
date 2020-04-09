import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import pickle
# from ai_sub2.doc.img_augmentation import img_aug
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from config import config
import os

BASE_DIR = os.path.join(config.base_dir, 'datasets')


def get_path_caption(caption_file_path):
    return np.loadtxt(caption_file_path, delimiter='|', skiprows=1, dtype=np.str)


def dataset_split_save(data, BASE_DIR, test_size=0.3):  # TODO config
    train_dataset, val_dataset = train_test_split(data,
                                                  test_size=test_size,
                                                  shuffle=False)

    np.save(os.path.join(BASE_DIR, 'train_datasets.npy'), train_dataset)
    np.save(os.path.join(BASE_DIR, 'test_datasets.npy'), val_dataset)


def get_data_file():
    train_datasets_path = os.path.join(BASE_DIR, 'train_datasets.npy')
    test_datasets_path = os.path.join(BASE_DIR, 'test_datasets.npy')
    dataset_path = train_datasets_path if config.do_what == 'train' else test_datasets_path
    data = np.load(os.path.join(BASE_DIR, dataset_path))
    if config.do_sampling:
        total_len = len(data)
        n_of_sample = int(total_len * config.do_sampling)
        img_paths = data[:n_of_sample, :1]
        captions = data[:n_of_sample, 2:]
    else:
        img_paths = data[:, :1]
        captions = data[:, 2:]
    train_images = np.squeeze(img_paths, axis=1)
    train_captions = np.squeeze(captions, axis=1)
    train_captions = ['<start>' + cap + ' <end>' for cap in train_captions]
    train_images, train_captions = shuffle(
        train_images, train_captions, random_state=1)
    return train_images, train_captions


def save_tokenizer(data_path, tokenizer_path, caption_num_words=5000):
    data = np.load(data_path)
    captions = data[:, 2:]

    captions = np.squeeze(captions, axis=1)
    captions = ['<start>' + cap + ' <end>' for cap in captions]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=caption_num_words + 1,
                                                      oov_token='<unk>',
                                                      lower=True,
                                                      split=' ',
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


def change_text_to_token(train_captions, tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
        train_seqs, padding='post')
    return cap_vector


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# def get_image_datasets(img_name_vector):
#     encode_train = sorted(set(img_name_vector))
#     image_dataset = list(map(load_image, encode_train))
#     # image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
#     # image_dataset = image_dataset.map(
#     #     load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)  # TODO batch
#     return image_dataset


def map_func(img_name, cap):
    feature_name = os.path.basename(img_name).decode('utf-8').replace('jpg', 'npy')
    img_tensor = np.load((os.path.join(BASE_DIR, 'features', feature_name)))
    return img_tensor, cap


# def get_tf_dataset(img_name_train, cap_train):
#     dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
#     dataset = dataset.map(lambda item1, item2: tf.numpy_function(
#         map_func, [item1, item2], [tf.float32, tf.int32]),
#         num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     return dataset
