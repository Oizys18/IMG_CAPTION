import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
from doc.img_augmentation import img_aug
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from config import config

def get_path_caption(caption_file_path):
    return np.loadtxt(caption_file_path, delimiter='|', skiprows=1, dtype=np.str)


def dataset_split_save(data, test_size=0.3): # TODO config
    train_dataset, val_dataset = train_test_split(data,
                                                  test_size=test_size,
                                                  shuffle=False)

    np.savetxt(
        './datasets/train_datasets.csv', train_dataset, fmt='%s', delimiter='|'
    )
    np.save('./datasets/train_datasets.npy', train_dataset)
    np.savetxt(
        './datasets/test_datasets.csv', val_dataset, fmt='%s', delimiter='|'
    )
    np.save('./datasets/test_datasets.npy', val_dataset)
    return './datasets/train_datasets.npy', './datasets/test_datasets.npy'


def get_data_file(dir_path, dataset_path):
    data = np.load(dataset_path)
    img_paths = data[:50, :1]
    captions = data[:50, 2:]
    train_images = np.squeeze(img_paths, axis=1)
    train_images = [dir_path + 'images/' + img for img in train_images]
    train_captions = np.squeeze(captions, axis=1)
    train_captions = ['<start>' + cap + ' <end>' for cap in train_captions]
    print()
    print('전처리 step 1')
    print('이미지 경로 수정 ex) ', train_images[:1])
    print('캡션 앞뒤 붙이기 ex) ')
    print('1: ', train_captions[:1])
    print('2: ', train_captions[1:2])
    return train_images, train_captions


def save_tokenizer(data_path, caption_num_words=5000):
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

    with open('./datasets/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


def change_text_to_token(train_captions):
    with open('./datasets/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    print()
    print('전처리 step 2')
    print('캡션 텍스트 토큰화 ex) ')
    print('1: ', cap_vector[:1])
    print('2: ', cap_vector[1:2])
    return cap_vector


def load_image(image_path):
    image = mpimg.imread(image_path).astype(np.uint8)

    # TODO 정규화 함수를 추가해주세요
    image = tf.image.per_image_standardization(image).numpy()

    # 데이터 증강
    image = img_aug(image,config.img_aug)
    # plt.imshow(image)
    # plt.show()
    
    return image, image_path


def get_image_datasets(img_name_vector):
    encode_train = sorted(set(img_name_vector))
    image_dataset = list(map(load_image,encode_train))
    # image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    # image_dataset = image_dataset.map(
    #     load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)  # TODO batch
    return image_dataset


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


def get_tf_dataset(img_name_train, cap_train):
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset
