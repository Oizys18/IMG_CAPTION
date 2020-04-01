import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle


# Req. 3-1	이미지 경로 및 캡션 불러오기
def get_path_caption(caption_file_path):
    return np.loadtxt(caption_file_path, delimiter='|', skiprows=1, dtype=np.str)


# Req. 3-2	전체 데이터셋을 분리해 저장하기
def dataset_split_save(data, test_size, random_state):
    train_dataset, val_dataset = train_test_split(data,
                                                  test_size=test_size,
                                                  shuffle=True,
                                                  random_state=random_state)

    np.savetxt(
        './datasets/train_datasets.csv', train_dataset, fmt='%s', delimiter='|'
    )
    np.savetxt(
        './datasets/val_datasets.csv', val_dataset, fmt='%s', delimiter='|'
    )

    return './datasets/train_datasets.csv', './datasets/val_datasets.csv'


# Req. 3-3	저장된 데이터셋 불러오기
def get_data_file(dataset_path):
    data = np.loadtxt(dataset_path, delimiter='|', dtype=np.str)
    img_paths = data[1:, :1]
    captions = data[1:, 2:]

    return img_paths, captions


# Req. 3-4	데이터 샘플링
def sampling_data(img_paths, captions, do_sampling):
    print('sampling 데이터를 ' + str(do_sampling) + '개 실행합니다.')

    return img_paths[:do_sampling, :], captions[:do_sampling, :]


# 2-2-1 Tokenizer 저장하기
def save_tokenizer(caption_file_path, caption_num_words=5000):
    data = np.loadtxt(caption_file_path, delimiter='|', dtype=np.str)
    captions = data[1:, 2:]

    captions = np.squeeze(captions, axis=1)
    captions = ['<start>' + cap + ' <end>' for cap in captions]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=caption_num_words + 3,
                                                      oov_token='<unk>',
                                                      lower=True,
                                                      split=' ',
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    with open('../datasets/tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


# 2-2-2 Tokenizer 불러오기
def get_tokenizer():
    with open('../datasets/tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


# 3-1 tf.data.Dataset 생성
def get_dataset(image_file_path, caption_file_path):
    pass
