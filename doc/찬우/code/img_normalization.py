import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# import tensorflow as tf



# img = Image.open('8251604257.jpg')
# img.thumbnail((400,400), Image.ANTIALIAS)
# plt.imshow(img)
# plt.show()

img2 = mpimg.imread('8251604257.jpg')
# print(img2)
plt.imshow(img2)
plt.show()
means = img2.mean(axis=(0,1,2))
print('----------------mean--------------')
print(means)
means.shape
print('----------------std--------------')
print(np.std(img2,axis=(0,1,2),keepdims=True))

# centered = abs((img2-np.mean(img2))/np.std(img2))
new = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
print(new)
# centered = (img2-img2.mean(axis=(0,1,2),keepdims=True))/np.std(img2,axis=(0,1,2),keepdims=True)
# print(centered)
# plt.imshow(centered)
plt.imshow(new)
plt.show()


# df = tf.image.per_image_standardization(img2)
# print(df)
# print(df/255)
# plt.imshow(img2)
# plt.show()
# print(img2)

