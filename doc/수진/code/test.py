# -*- coding: utf-8 -*-
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import PIL 
import tensorflow as tf

# 이미지 불러오기 
# C:\\Users\\multicampus\\ai_sub2\\datasets\\train_datasets.csv
def fileload(images_path):
    # train_dataset 에서 이미지 파일 경로를  불러온다. 
    data = np.loadtxt(images_path, delimiter='|', dtype=np.str)
    img_paths = data[:, :1]
    
    for path in img_paths:
        
    # 이미지 불러오기
    img = PIL.Image.open(f'C:\\Users\\multicampus\\ai_sub2\\datasets\\images\\36979.jpg')
    result = img.resize((64, 64))
    return result

def nomalization(All_img):
    # R, G, B 채널별로 장당 정규화 
    scaler = preprocessing.StandardScaler().fit(All_img)
        
    
    
    
    
    
 