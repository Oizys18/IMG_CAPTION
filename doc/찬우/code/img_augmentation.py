import matplotlib.image as mpimg
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from config import config
from pathlib import Path


def img_aug(image, method):
    # image = mpimg.imread(
    #     Path(config.base_dir, 'datasets', 'images', img_path))
    # print('original image')
    # ia.imshow(image)    
    
    # sequential
    if method == 'seq':
        aug = iaa.Sequential([
            iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
            iaa.AddToHueAndSaturation((-60, 60)),
            iaa.ElasticTransformation(alpha=90, sigma=9), 
            iaa.Cutout(),
            
        ], random_order=True)

        image = aug(image=image)
        # print("water-like")
        # ia.imshow(image)

    # blend alpha
    elif method == 'ba':
        aug = iaa.BlendAlphaVerticalLinearGradient(
            iaa.AveragePooling(11),
            start_at=(0.0, 1.0), end_at=(0.0, 1.0))
        image = aug(image=image)
        # print('blend alpha')
        # ia.imshow(image)
    
    # mosaic
    elif method == 'ba_box':
        aug = iaa.BlendAlphaRegularGrid(nb_rows=2, nb_cols=2,
                                 foreground=iaa.Multiply(0.0),
                                 background=iaa.AveragePooling(8),
                                 alpha=[0.0, 0.0, 1.0])
        image = aug(image=image)
        # print('blend_alpha with boxes')
        # ia.imshow(image)
    return image


# # imageio 라이브러리 사용
# img_path = '8086165175.jpg'
# image = imageio.imread('../../../datasets/images/' + img_path)
# print('original image')
# ia.imshow(image)

# # sequential
# seq = iaa.Sequential([
#     # crop and pad images
#     iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
#     iaa.AddToHueAndSaturation((-60, 60)),  # change their color
#     iaa.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
#     iaa.Cutout()  # replace one squared area within the image by a constant intensity value
# ], random_order=True)

# images_aug = seq(image=image)
# print("Augmented:")
# ia.imshow(images_aug)


# # blendalpha 모자이크
# aug = iaa.BlendAlphaVerticalLinearGradient(
#     iaa.AveragePooling(11),
#     start_at=(0.0, 1.0), end_at=(0.0, 1.0))

# image = aug(image=image)
# print('blendalpha')
# ia.imshow(image)


# # blendalpha regular grid , 검은상자 만들기
# aug2 = iaa.BlendAlphaRegularGrid(nb_rows=(4, 6), nb_cols=(1, 4),
#                                  foreground=iaa.Multiply(0.0))

# image2 = aug2(image=image)
# print('regular grid')
# ia.imshow(image2)


# # 랜덤검은상자 + 모자이크 테스트
# aug3 = iaa.BlendAlphaRegularGrid(nb_rows=2, nb_cols=2,
#                                  foreground=iaa.Multiply(0.0),
#                                  background=iaa.AveragePooling(8),
#                                  alpha=[0.0, 0.0, 1.0])

# image3 = aug3(image=image)
# print('regular grid')
# ia.imshow(image3)
