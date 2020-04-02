from config import config
from data import preprocess
from utils import utils


# config 저장
utils.save_config()

# 이미지 경로 및 캡션 불러오기
# dataset = preprocess.get_path_caption(config.caption_file_path)

# 전체 데이터셋을 분리해 저장하기
# train_dataset_path, test_dataset_path = preprocess.dataset_split_save(dataset, config.test_size, config.random_state)

# tokenizer 만들기
# preprocess.save_tokenizer(train_dataset_path)


# 저장된 데이터셋 불러오기
train_dataset_path = './datasets/train_datasets.npy'
test_dataset_path = './datasets/test_datasets.npy'
dataset_path = train_dataset_path if config.do_what == 'train' else test_dataset_path
img_paths, caption = preprocess.get_data_file(dataset_path)


# 데이터 샘플링
if config.do_sampling:
    img_paths, caption = preprocess.sampling_data(img_paths, caption, config.do_sampling)


# 이미지와 캡션 시각화 하기
utils.visualize_img_caption(img_paths, caption)
