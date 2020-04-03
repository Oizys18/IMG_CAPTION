import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


def get_data_file():
    data = np.load('./datasets/test_datasets.npy')
    img_paths = data[:50, :1]
    captions = data[:50, 2:]
    train_images = np.squeeze(img_paths, axis=1)
    train_images = ['./datasets/images/' + img for img in train_images]
    train_captions = np.squeeze(captions, axis=1)
    train_captions = ['<start>' + cap + ' <end>' for cap in train_captions]
    train_images = list(set(train_images))  # 테스트를 위한 중복제거
    print(train_images[:3])
    print(train_captions[:3])
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


# tensorflow 사용, 솔지 코드
def img_normalization_2(img):
    # tf_img = tf.constant(img, dtype=tf.float32)
    mean, var = tf.nn.moments(img, axes=[0, 1, 2])
    nn_moments_image = (img - mean) / var**0.5
    image_standardization_image = tf.image.per_image_standardization(img)
    return [nn_moments_image, image_standardization_image]


train_images, train_captions = get_data_file()

train_images = train_images[:2]
for train_image in train_images:
    img, image_path = image_load(train_image)
    images1 = img_normalization_1(image_path)
    images2 = img_normalization_2(img)
    titles = ['origin_img', 'min_max_image', 'mean_std_image', 'nn_moments_image', 'image_standardization_image']
    images = images1 + images2
    f = plt.figure()
    for i, image in enumerate(images):
        f.add_subplot(2, 3, i+1)
        plt.title(titles[i])
        plt.imshow(image)
    plt.show()