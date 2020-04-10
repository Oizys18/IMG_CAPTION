from imgaug import augmenters as iaa
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from config import config
from pathlib import Path
import tensorflow as tf
import imgaug as ia
import random


def img_aug(img_name):
    image_string = tf.io.read_file(
        str(Path(config.base_dir, 'datasets', 'images', img_name)))
    image = tf.image.decode_jpeg(image_string, channels=3)
    ran = random.randint(1, 5)
    if ran == 1:
        image = tf.image.random_flip_left_right(image)
    elif ran == 2:
        image = tf.image.rot90(image)
    elif ran == 3:
        image = tf.image.random_flip_up_down(image)
    elif ran == 4:
        image = tf.image.adjust_brightness(image, 0.4)
    elif ran == 5:
        image = tf.image.central_crop(image, central_fraction=0.5)
    return image.numpy()
