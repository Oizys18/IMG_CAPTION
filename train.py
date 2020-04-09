from config import config
from data import preprocess
from utils import utils
from data.feature_extraction import feature_extraction
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import numpy as np
from models.encoder import CNN_Encoder
from models.decoder import RNN_Decoder

tf.autograph.experimental.do_not_convert()
tf.compat.v1.reset_default_graph()
# config 저장
utils.save_config()
BASE_DIR = os.path.join(config.base_dir, 'datasets')

# 전체 데이터셋을 분리해 저장하기
train_datasets_path = os.path.join(BASE_DIR, 'train_datasets.npy')
test_datasets_path = os.path.join(BASE_DIR, 'test_datasets.npy')
if not os.path.exists(train_datasets_path):
    # 이미지 경로 및 캡션 불러오기
    dataset = preprocess.get_path_caption(config.caption_file_path)
    preprocess.dataset_split_save(dataset, config.test_size)
    print('dataset 을 train_datasets 과 test_datasets 으로 나눕니다.')
else:
    print('저장 된 train_datasets 과 test_datasets 을 사용합니다.')

# tokenizer 만들기
tokenizer_path = os.path.join(BASE_DIR,'tokenizer.pkl')
if not os.path.exists(tokenizer_path):
    preprocess.save_tokenizer(train_datasets_path)
    print('새로운 Tokenizer 를 저장합니다.')
else:
    print('기존의 Tokenizer 를 사용합니다.')



# feature extraction 이미지 특징 생성하기
feature_extraction()


# 이미지 특징 벡터 불러오기
# Load the numpy files
def map_func(img_name, cap):
    feature_name = os.path.basename(img_name).decode('utf-8').replace('jpg', 'npy')
    img_tensor = np.load((os.path.join(BASE_DIR, 'features', feature_name)))
    return img_tensor, cap




# 전처리 7 Req 3-2


# hyperparameter
embedding_dim = 256
BUFFER_SIZE = 48
BATCH_SIZE = 16
units = 512
vocab_size = 5000


################################################ 실행 여기서 부터
# file load 
img_name_vector, train_captions = preprocess.get_data_file()
cap_vector = preprocess.change_text_to_token(train_captions)
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=config.test_size,
                                                                    random_state=config.random_state)  

# 학습을 위한 데이터셋 설정 (tf.data dataset)
tf.compat.v1.enable_eager_execution()
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)



# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# for x, y in dataset:
#     print(x, y)


encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

########### train
for (batch, (img_tensor, target)) in enumerate(dataset):
    # print(np.shape(img_tensor)) 
    # (16, 64, 2048)
    features = encoder(img_tensor)
    # print(np.shape(features))
    # (16, 64, 256)

