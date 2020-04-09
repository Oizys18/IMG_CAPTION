import matplotlib.image as mpimg
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
from config import config
from pathlib import Path
import matplotlib.pyplot as plt
import random
img_names=['36979.jpg','256063.jpg','371902.jpg']

def img_aug(img_name, method):
    # tf
    # image_string=tf.io.read_file(str(Path(config.base_dir, 'datasets', 'images', img_name)))
    # image=tf.image.decode_jpeg(image_string,channels=3)
    # ran = random.randint(1,5)
    # if ran == 1:
    #     image = tf.image.random_flip_left_right(image)
    # elif ran == 2:
    #     image = tf.image.rot90(image)
    # elif ran == 3:
    #     image = tf.image.random_flip_up_down(image)
    # elif ran == 4:
    #     image = tf.image.adjust_brightness(image, 0.4)
    # elif ran == 5:
    #     image = tf.image.central_crop(image, central_fraction=0.5)

    image = mpimg.imread(
        Path(config.base_dir, 'datasets', 'images', img_name))
    # sequential
    if method == 'seq':
        aug = iaa.Sequential([
            iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
            iaa.AddToHueAndSaturation((-60, 60)),
            iaa.ElasticTransformation(alpha=90, sigma=9),
            iaa.Cutout()
        ], random_order=True)
        image = aug(image=image)

    # blend alpha
    elif method == 'ba':
        aug = iaa.BlendAlphaVerticalLinearGradient(
            iaa.AveragePooling(11),
            start_at=(0.0, 1.0), end_at=(0.0, 1.0))
        image = aug(image=image)
    elif method == 'flip_h':
        aug = iaa.fliplr(
            iaa.Fliplr(0.5),  # horizontal flips
        )
        image = aug(image=image)
    elif method == 'flip_v':
        aug = iaa.flipud(
            iaa.Fliplr(0.5),  # vertical flips
        )
        image = aug(image=image)

    # mosaic
    elif method == 'ba_box':
        aug = iaa.BlendAlphaRegularGrid(nb_rows=2, nb_cols=2,
                                        foreground=iaa.Multiply(0.0),
                                        background=iaa.AveragePooling(8),
                                        alpha=[0.0, 0.0, 1.0])
        image = aug(image=image)

    return image

for img_name in img_names:
    image = img_aug(img_name)
    plt.imshow(image)
    plt.show()