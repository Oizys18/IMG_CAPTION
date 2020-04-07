from config import config
from data import preprocess
from utils import utils
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

# config 저장
utils.save_config()

# 나중에 merge 이후로 config 에서 가져오기
PATH = os.path.abspath('.') + '/datasets/'

# 전체 데이터셋을 분리해 저장하기
train_datasets_path = PATH + 'train_datasets.npy'
test_datasets_path = PATH + 'test_datasets.npy'
if not os.path.exists(train_datasets_path):
    # 이미지 경로 및 캡션 불러오기
    dataset = preprocess.get_path_caption(config.caption_file_path)
    preprocess.dataset_split_save(dataset, config.test_size)
    print('dataset 을 train_datasets 과 test_datasets 으로 나눕니다.')
else:
    print('저장 된 train_datasets 과 test_datasets 을 사용합니다.')

# tokenizer 만들기
tokenizer_path = PATH + 'tokenizer.pkl'
if not os.path.exists(tokenizer_path):
    preprocess.save_tokenizer(train_datasets_path)
    print('새로운 Tokenizer 를 저장합니다.')
else:
    print('기존의 Tokenizer 를 사용합니다.')

# 전처리 1 image_path 와 captions 데이터셋 불러오기
dataset_path = train_datasets_path if config.do_what == 'train' else test_datasets_path
img_name_vector, train_captions = preprocess.get_data_file(PATH, dataset_path)

# 전처리 2 captions 의 text 를 토큰화
cap_vector = preprocess.change_text_to_token(train_captions)

# 전처리 3 train 과 val 분리
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)  # TODO config

# 전처리 4 image 불러와서 정규화 하고 tf.data.datasets 으로 return
# TODO 정규화 함수 추가 해야 합니다.
# TODO 이 dataset 으로 수진이 pre_trained 모델 사용할 것
image_dataset = preprocess.get_image_datasets(img_name_vector)

# 전처리 5 pre_trained 모델을 통해서 특징 npy 파일로 저장하기


# 전처리 6 Req 3-1 이미지 데이터 및 토큰화 된 캡션 쌍 리턴하기 및 Shuffle, batch
# dataset = preprocess.get_tf_dataset(img_name_train, cap_train)
# dataset = dataset.shuffle(config.buffer_size).batch(config.batch_size)  # TODO config
# dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 전처리 7 Req 3-2
