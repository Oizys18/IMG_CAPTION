import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle


def get_data_file():
    data = np.load('../../../datasets/test_datasets.npy')
    img_paths = data[:50, :1]
    captions = data[:50, 2:]
    train_images = np.squeeze(img_paths, axis=1)
    train_images = ['../../../datasets/images/' + img for img in train_images]
    train_captions = np.squeeze(captions, axis=1)
    train_captions = ['<start>' + cap + ' <end>' for cap in train_captions]
    return train_images, train_captions


def image_load(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (255, 255))
    return img, image_path


# numpy 사용, 찬우 코드
def img_normalization_1(image_path):
    img = Image.open(image_path)
    img = img.resize((255, 255))
    img2 = np.array(img)
    min_max_image = (img - np.min(img)) / (np.max(img) - np.min(img))
    mean_std_image = (img-img2.mean(axis=(0,1,2),keepdims=True))/np.std(img,axis=(0,1,2),keepdims=True)
    return [img, min_max_image, mean_std_image]


def change_text_to_token(train_captions):
    with open('../../../datasets/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    return cap_vector


img_name_vector, train_captions = get_data_file()
encode_train = sorted(set(img_name_vector))
cap_vector = change_text_to_token(train_captions)

# Create training and validation sets using an 80-20 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)

print('=========Create training and validation sets==============')
print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))

dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))


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
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
