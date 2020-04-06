import tensorflow as tf

from sklearn.model_selection import train_test_split

import numpy as np
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

    print(train_images[:3])
    print(train_captions[:3])
    return train_images, train_captions


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# 이미지 전처리 모델 정의
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


img_name_vector, train_captions = get_data_file()
# Get unique images
encode_train = sorted(set(img_name_vector))
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
# print(image_dataset)
# <TensorSliceDataset shapes: (), types: tf.string>
# print(image_dataset.enumerate())
# <DatasetV1Adapter shapes: ((), ()), types: (tf.int64, tf.string)>
print('------------image_dataset---------------------')
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
        # print(path_of_feature)
        # print(bf.numpy())
        np.save(path_of_feature, bf.numpy())

"""
print(batch_features)
tf.Tensor(
[[[0.         0.42186102 0.         ... 0.         0.         0.38468894]
  [0.         0.04957559 0.         ... 0.         0.         0.17241111]
  [0.17579332 0.         0.13364312 ... 0.         0.         0.18271159]
  ...
  [0.         0.         0.         ... 0.         0.         0.        ]
  [0.         0.         0.         ... 0.         0.         0.        ]
  [0.         0.         0.         ... 0.         0.         0.25328407]]

 [[1.578953   0.         0.         ... 0.27585867 0.         0.        ]
  [2.0826812  0.         0.         ... 0.         0.         0.        ]
  [2.389288   0.         0.         ... 0.         0.         0.01047712]
  ...
  [0.         0.         0.         ... 0.         0.         0.        ]
  [0.24908888 0.         0.25783932 ... 0.         0.         0.        ]
  [0.2582202  0.         0.3799192  ... 0.         0.         0.        ]]], shape=(2, 64, 2048), dtype=float32)

print(p)
tf.Tensor(b'../../../datasets/images/4690435409.jpg', shape=(), dtype=string)

print(path_of_feature)
../../../datasets/images/4690435409.jpg

print(bf.numpy())
[[0.         0.42186102 0.         ... 0.         0.         0.38468894]
 [0.         0.04957559 0.         ... 0.         0.         0.17241111]
 [0.17579332 0.         0.13364312 ... 0.         0.         0.18271159]
 ...
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.25328407]]
"""


# Find the maximum length of any caption in our dataset
# def calc_max_length(tensor):
#     return max(len(t) for t in tensor)

# top_k = 5000
# tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
#                                                   oov_token="<unk>",
#                                                   filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
# tokenizer.fit_on_texts(train_captions)
# tokenizer.word_index['<pad>'] = 0
# tokenizer.index_word[0] = '<pad>'


def get_tokenizer():
    with open('../../../datasets/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


tokenizer = get_tokenizer()
train_seqs = tokenizer.texts_to_sequences(train_captions)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
# max_length = calc_max_length(train_seqs)


# Create training and validation sets using an 80-20 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)

print(
    '=======================================Create training and validation sets=======================================')
print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))
# 400 400 100 100
print()

# Feel free to change these parameters according to your system's configuration
BATCH_SIZE = 64
# 데이터셋을 섞을 버퍼 크기
# (TF 데이터는 무한한 시퀀스와 함께 작동이 가능하도록 설계되었으며,
# 따라서 전체 시퀀스를 메모리에 섞지 않습니다. 대신에, 요소를 섞는 버퍼를 유지합니다).
BUFFER_SIZE = 1000
# 임베딩 차원
embedding_dim = 256
# RNN 유닛(unit) 개수
units = 512
# 문자로 된 어휘 사전의 크기
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64


# Load the numpy files
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
    map_func, [item1, item2], [tf.float32, tf.int32]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)