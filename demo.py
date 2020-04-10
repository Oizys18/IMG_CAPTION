from config import config
import tensorflow as tf
import predict
from PIL import Image
from tkinter import *
from tkinter import filedialog

root = Tk()
root.filename = filedialog.askopenfilename(
    initialdir="E:/Images", title="이미지 파일을 선택하세요!")
print(root.filename)
root.withdraw()

image_path = root.filename
result, attention_plot = predict.evaluate(image_path)
print ('Prediction Caption:', ' '.join(result))
predict.plot_attention(image_path, result, attention_plot)
# opening the image
# Image.open(image_path)
