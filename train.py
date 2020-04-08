from config import config
from data import preprocess
from utils import utils
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

# config 저장
print(config.do_sampling)
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

# 전처리 1 image_path 와 captions 데이터셋 불러오기
img_name_vector, train_captions = preprocess.get_data_file()


'''

# 전처리 2 captions 의 text 를 토큰화
cap_vector = preprocess.change_text_to_token(train_captions)


# 전처리 3 train 과 val 분리
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)  # TODO config


# 이미지 특징 벡터 불러오기
# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap



# 인코더
class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


# 전처리 6 Req 3-1 이미지 데이터 및 토큰화 된 캡션 쌍 리턴하기 및 Shuffle, batch
# dataset = preprocess.get_tf_dataset(img_name_train, cap_train)
# dataset = dataset.shuffle(config.buffer_size).batch(config.batch_size)  # TODO config
# dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 전처리 7 Req 3-2


# hyperparameter
embedding_dim = 256



# 실행 여기서 부터
encoder = CNN_Encoder(embedding_dim)
features = encoder(img_tensor)
'''