from config import config
from data import preprocess
from utils import utils
import os

# config 저장
utils.save_config()

# 나중에 merge 이후로 config 에서 가져오기
PATH = os.path.abspath('.') + '/datasets/'

# 전체 데이터셋을 분리해 저장하기
datasets_path = 'train_datasets.npy'
if not os.path.exists(PATH + datasets_path):
    # 이미지 경로 및 캡션 불러오기
    dataset = preprocess.get_path_caption(config.caption_file_path)
    preprocess.dataset_split_save(dataset, config.test_size, config.random_state)
    print('dataset 을 train_datasets 과 test_datasets 으로 나눕니다.')
else:
    print('주어진 train_datasets 과 test_datasets 을 사용합니다.')

# tokenizer 만들기
tokenizer_path = 'tokenizer.pkl'
if not os.path.exists(PATH + tokenizer_path):
    preprocess.save_tokenizer(PATH + datasets_path)
    print('새로운 Tokenizer 를 저장합니다.')

# 저장된 데이터셋 불러오기
# train_dataset_path = 'train_datasets.npy'
# test_dataset_path = 'test_datasets.npy'
# dataset_path = train_dataset_path if config.do_what == 'train' else test_dataset_path
# img_paths, caption = preprocess.get_data_file(PATH + dataset_path)


# # 데이터 샘플링
# if config.do_sampling:
#     img_paths, caption = preprocess.sampling_data(img_paths, caption, config.do_sampling)
#
#
# # 이미지와 캡션 시각화 하기
# utils.visualize_img_caption(img_paths, caption)