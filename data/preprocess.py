import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle


def get_path_caption(caption_file_path):
    return np.loadtxt(caption_file_path, delimiter='|', skiprows=1, dtype=np.str)


def dataset_split_save(data, test_size, random_state):
    train_dataset, val_dataset = train_test_split(data,
                                                  test_size=0.3,
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


def get_data_file(dataset_path):
    data = np.load(dataset_path)
    img_paths = data[:, :1]
    captions = data[:, 2:]

    return img_paths, captions


def sampling_data(img_paths, captions, do_sampling):
    print('sampling 데이터를 ' + str(do_sampling) + '개 실행합니다.')

    return img_paths[:do_sampling, :], captions[:do_sampling, :]


def save_tokenizer(data_path, caption_num_words=5000):
    data = np.loadtxt('./datasets/train_datasets.csv', delimiter='|', skiprows=1, dtype=np.str)
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
    print(tokenizer.index_word)
    with open('./datasets/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_tokenizer():
    with open('./datasets/tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def get_dataset(image_file_path, caption_file_path):
    pass



