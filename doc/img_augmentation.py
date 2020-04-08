import matplotlib.image as mpimg
import imgaug as ia
from imgaug import augmenters as iaa
from config import config
from pathlib import Path


def img_aug(img_name, method):
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

    # mosaic
    elif method == 'ba_box':
        aug = iaa.BlendAlphaRegularGrid(nb_rows=2, nb_cols=2,
                                        foreground=iaa.Multiply(0.0),
                                        background=iaa.AveragePooling(8),
                                        alpha=[0.0, 0.0, 1.0])
        image = aug(image=image)

    return image
