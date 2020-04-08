import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from tqdm import tqdm

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
# config.gpu_options.allow_growth = True
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



def get_random_caption_tokens(idx):
    """
    Given a list of indices for images in the training-set,
    select a token-sequence for a random caption,
    and return a list of all these token-sequences.
    """

    # Initialize an empty list for the results.
    result = []

    # For each of the indices.
    for i in idx:
        # The index i points to an image in the training-set.
        # Each image in the training-set has at least 5 captions
        # which have been converted to tokens in tokens_train.
        # We want to select one of these token-sequences at random.

        # Get a random index for a token-sequence.
        j = np.random.choice(len(tokens_train[i]))

        # Get the j'th token-sequence for image i.
        tokens = tokens_train[i][j]

        # Add this token-sequence to the list of results.
        result.append(tokens)

    return result


def batch_generator(batch_size):
    """
    Generator function for creating random batches of training-data.

    Note that it selects the data completely randomly for each
    batch, corresponding to sampling of the training-set with
    replacement. This means it is possible to sample the same
    data multiple times within a single epoch - and it is also
    possible that some data is not sampled at all within an epoch.
    However, all the data should be unique within a single batch.
    """

    # Infinite loop.
    while True:
        # Get a list of random indices for images in the training-set.
        idx = np.random.randint(num_images_train,
                                size=batch_size)

        # Get the pre-computed transfer-values for those images.
        # These are the outputs of the pre-trained image-model.
        transfer_values = transfer_values_train[idx]

        # For each of the randomly chosen images there are
        # at least 5 captions describing the contents of the image.
        # Select one of those captions at random and get the
        # associated sequence of integer-tokens.
        tokens = get_random_caption_tokens(idx)

        # Count the number of tokens in all these token-sequences.
        num_tokens = [len(t) for t in tokens]

        # Max number of tokens.
        max_tokens = np.max(num_tokens)

        # Pad all the other token-sequences with zeros
        # so they all have the same length and can be
        # input to the neural network as a numpy array.
        tokens_padded = pad_sequences(tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')

        # Further prepare the token-sequences.
        # The decoder-part of the neural network
        # will try to map the token-sequences to
        # themselves shifted one time-step.
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        # Dict for the input-data. Because we have
        # several inputs, we use a named dict to
        # ensure that the data is assigned correctly.
        x_data = \
            {
                'decoder_input': decoder_input_data,
                'transfer_values_input': transfer_values
            }

        # Dict for the output-data.
        y_data = \
            {
                'decoder_output': decoder_output_data
            }

        yield (x_data, y_data)



def get_data_file():
    data = np.load('../../../datasets/test_datasets.npy')
    img_paths = data[:50, :1]
    captions = data[:50, 2:]
    train_images = np.squeeze(img_paths, axis=1)
    train_images = ['../../../datasets/images/' + img for img in train_images]
    train_captions = np.squeeze(captions, axis=1)
    train_captions = ['<start>' + cap + ' <end>' for cap in train_captions]


    captions_pack = [train_captions[i*5:(i+1)*5] for i in range(10)]

    print(train_images[:3])
    print(train_captions[:3])
    return train_images, train_captions, captions_pack


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


img_name_vector, train_captions, captions_pack = get_data_file()

# image_model = tf.keras.applications.InceptionV3(include_top=False,
#                                                 weights='imagenet')
# new_input = image_model.input
# hidden_layer = image_model.layers[-1].output
# image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
#
# # Get unique images
# encode_train = sorted(set(img_name_vector))
#
# # Feel free to change batch_size according to your system configuration
# image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
# # print(image_dataset)
# # <TensorSliceDataset shapes: (), types: tf.string>
# # print(image_dataset.enumerate())
# # <DatasetV1Adapter shapes: ((), ()), types: (tf.int64, tf.string)>
# print('------------image_dataset---------------------')
# image_dataset = image_dataset.map(
#     load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)
#
# print(image_dataset)
# # <BatchDataset shapes: ((None, 299, 299, 3), (None,)), types: (tf.float32, tf.string)>
# for img, path in tqdm(image_dataset):
#     batch_features = image_features_extract_model(img)
#
#     batch_features = tf.reshape(batch_features,
#                                 (batch_features.shape[0], -1, batch_features.shape[3]))
#
#     for bf, p in zip(batch_features, path):
#         path_of_feature = p.numpy().decode("utf-8")
#         np.save(path_of_feature, bf.numpy())
#
#
# # Find the maximum length of any caption in our dataset
# def calc_max_length(tensor):
#     return max(len(t) for t in tensor)
#
#
# # top_k = 5000
# def get_tokenizer():
#     with open('../../../datasets/tokenizer.pkl', 'rb') as f:
#         tokenizer = pickle.load(f)
#     return tokenizer
#
#
# tokenizer = get_tokenizer()
# train_seqs = tokenizer.texts_to_sequences(train_captions)
# cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
# max_length = calc_max_length(train_seqs)
#
# # Create training and validation sets using an 80-20 split
# img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
#                                                                     cap_vector,
#                                                                     test_size=0.2,
#                                                                     random_state=0)
#
# print(
#     '=======================================Create training and validation sets=======================================')
# print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))
# # 400 400 100 100
# print()
#
# # https://towardsdatascience.com/practical-coding-in-tensorflow-2-0-fafd2d3863f6
# #
# # Feel free to change these parameters according to your system's configuration
# BATCH_SIZE = 64
# # 데이터셋을 섞을 버퍼 크기
# # (TF 데이터는 무한한 시퀀스와 함께 작동이 가능하도록 설계되었으며,
# # 따라서 전체 시퀀스를 메모리에 섞지 않습니다. 대신에, 요소를 섞는 버퍼를 유지합니다).
# BUFFER_SIZE = 10000
# # 임베딩 차원
# embedding_dim = 256
# # RNN 유닛(unit) 개수
# units = 512
# # 문자로 된 어휘 사전의 크기
# # 케라스 토크나이저의 정수 인코딩은 인덱스가 1부터 시작하지만,
# # 케라스 원-핫 인코딩에서 배열의 인덱스가 0부터 시작하기 때문에
# # 배열의 크기를 실제 단어 집합의 크기보다 +1로 생성해야하므로 미리 +1 선언
# vocab_size = len(tokenizer.word_index) + 1
# num_steps = len(img_name_train) // BATCH_SIZE
# # Shape of the vector extracted from InceptionV3 is (64, 2048)
# # These two variables represent that vector shape
# features_shape = 2048
# attention_features_shape = 64
#
#
# # Load the numpy files
# def map_func(img_name, cap):
#     img_tensor = np.load(img_name.decode('utf-8') + '.npy')
#     return img_tensor, cap
#
#
# dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
#
# # Use map to load the numpy files in parallel
# dataset = dataset.map(lambda item1, item2: tf.numpy_function(
#     map_func, [item1, item2], [tf.float32, tf.int32]),
#                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
# # Shuffle and batch
# dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#
