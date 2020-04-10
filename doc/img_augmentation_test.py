import matplotlib.image as mpimg
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
from pathlib import Path
import matplotlib.pyplot as plt
import random

def img_aug(img_name):
    image = mpimg.imread(
        Path('..', 'datasets', 'images', img_name))

    ran = random.randint(1,5)
    if ran == 1:
        aug = iaa.Sequential([
            iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
            iaa.AddToHueAndSaturation((-60, 60)),
            iaa.ElasticTransformation(alpha=90, sigma=9),
            iaa.Cutout()
        ], random_order=True)
        image = aug(image=image)
    elif ran == 2:
        aug = iaa.BlendAlphaRegularGrid(nb_rows=2, nb_cols=2,
                                        foreground=iaa.Multiply(0.0),
                                        background=iaa.AveragePooling(8),
                                        alpha=[0.0, 0.0, 1.0])
        image = aug(image=image)
        
    elif ran == 3:
        image = tf.image.adjust_brightness(image, 0.4)
    elif ran == 4:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.rot90(image)
        image = tf.image.random_flip_up_down(image)
    elif ran == 5:    
        image = tf.image.central_crop(image, central_fraction=0.5)
    return image



img_names=['36979.jpg','256063.jpg','371902.jpg']
for image_name in img_names:
    image = img_aug(image_name)
    plt.imshow(image)
    plt.show()