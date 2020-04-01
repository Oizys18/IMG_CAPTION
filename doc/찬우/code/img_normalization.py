# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image


img = Image.open('8251604257.jpg')
img.thumbnail((250,250), Image.ANTIALIAS)
img.thumbnail()
plt.imshow(img)
plt.show()
