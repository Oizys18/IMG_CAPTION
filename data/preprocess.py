import numpy as np
from sklearn.model_selection import train_test_split


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
    img_paths = data[:, :1]
    captions = data[:, 2:]

    return img_paths, captions


# Req. 3-4	데이터 샘플링
def sampling_data(img_paths, captions, do_sampling):
    print('sampling 데이터를 ' + str(do_sampling) + '개 실행합니다.')

    return img_paths[:do_sampling, :], captions[:do_sampling, :]