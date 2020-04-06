import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


def get_data_file():
    data = np.load('../../datasets/test_datasets.npy')
    img_paths = data[:50, :1]
    captions = data[:50, 2:]
    train_images = np.squeeze(img_paths, axis=1)
    train_images = ['../../datasets/images/' + img for img in train_images]
    train_captions = np.squeeze(captions, axis=1)
    train_captions = ['<start>' + cap + ' <end>' for cap in train_captions]
    # train_images = list(set(train_images))  # 테스트를 위한 중복제거
    return train_images, train_captions


def image_load(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (255, 255))
    return img, image_path


# tensorflow 사용, 솔지 코드
def img_normalization_2(img):
    image_standardization_image = tf.image.per_image_standardization(img)
    return image_standardization_image


train_images, train_captions = get_data_file()

# for train_image in train_images:
#     img, image_path = image_load(train_image)
#     image = img_normalization_2(img)
#     plt.imshow(image)
#     plt.show()



for train_image in train_images:
    img, image_path = image_load(train_image)
    image = img_normalization_2(img)
    # print(image)

np.save('./test', image)

# test = np.load('.\\test.npy')
# print(test)