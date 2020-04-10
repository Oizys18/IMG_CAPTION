import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import pickle
import os


def get_path_caption(caption_file_path):
    return np.loadtxt(caption_file_path, delimiter='|', skiprows=1, dtype=np.str)


def dataset_split_save(base_dir, caption_file_path, test_size):
    train_datasets_path = os.path.join(base_dir, 'train_datasets.npy')
    test_datasets_path = os.path.join(base_dir, 'test_datasets.npy')
    if not os.path.exists(train_datasets_path):
        dataset = get_path_caption(caption_file_path)
        train_dataset, val_dataset = train_test_split(dataset,
                                                      test_size=test_size,
                                                      shuffle=False)
        np.save(train_datasets_path, train_dataset)
        np.save(test_datasets_path, val_dataset)
        print('dataset 을 train_datasets 과 test_datasets 으로 나눕니다.')
    else:
        print('저장 된 train_datasets 과 test_datasets 을 사용합니다.')


def get_data_file(base_dir, do_what, do_sampling):
    train_datasets_path = os.path.join(base_dir, 'train_datasets.npy')
    test_datasets_path = os.path.join(base_dir, 'test_datasets.npy')
    dataset_path = train_datasets_path if do_what == 'train' else test_datasets_path
    data = np.load(os.path.join(base_dir, dataset_path))
    if do_sampling:
        total_len = len(data)
        n_of_sample = int(total_len * do_sampling)
        img_paths = data[:n_of_sample, :1]
        captions = data[:n_of_sample, 2:]
    else:
        img_paths = data[:, :1]
        captions = data[:, 2:]
    train_images = np.squeeze(img_paths, axis=1)
    train_captions = np.squeeze(captions, axis=1)
    train_captions = ['<start>' + cap + ' <end>' for cap in train_captions]
    train_images, train_captions = shuffle(train_images, train_captions, random_state=1)
    return train_images, train_captions


def get_tokenizer(tokenizer_path, caption_file_path, num_words):
    if not os.path.exists(tokenizer_path):
        dataset = get_path_caption(caption_file_path)
        captions = dataset[:, 2:]

        captions = np.squeeze(captions, axis=1)
        captions = ['<start>' + cap + ' <end>' for cap in captions]

        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words + 1,
                                                          oov_token='<unk>',
                                                          lower=True,
                                                          split=' ',
                                                          filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

        tokenizer.fit_on_texts(captions)
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'

        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)

    return tokenizer


def change_text_to_token(tokenizer_path, train_captions):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    max_length = max(len(t) for t in train_seqs)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    return cap_vector, max_length


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


