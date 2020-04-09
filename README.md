# 🤞🥰 이미지 캡셔닝 기능 구현

![](https://img.shields.io/badge/version-2.0.0-green.svg) ![](https://img.shields.io/badge/created__at-20.04.03-yellow.svg) ![](https://img.shields.io/badge/updated__at-20.04.10-blue.svg)

> 인공지능 프로젝트 "이미지 캡셔닝 활용 시스템" Sub PJT 2



-----

## 🛒 Table of Contents

- [Installation](#installation)
- [Quick Start](#Quick Start)
- [Features](#features)
- [Documentation](#Documentation)
- [Team](#Team)

---------





## 🏃‍♀️🏃‍♂️ Installation 

### Clone

- 다음 링크를 통해 리포를 클론합니다.  
- HTTPS `https://lab.ssafy.com/s02-ai-sub2/s02p22a405.git`

### Setup

- 이미지 파일을 [다운로드](https://i02lab1.p.ssafy.io/) 하고 `datasets/` 에 위치시킵니다.

- 가상환경을 설정해줍니다. 프로젝트는 패키지 관리 및 가상환경 설정을 위해 Anaconda 를 사용합니다.

  ```bash
  conda env create -f AI.yaml
  ```

  - 또는 다음 [파일](.\doc\spec-file.txt)을 참고하여 프로젝트에서 사용 된 프로그램을 확인, 설치할 수 있습니다. 

  - 기본적인 프로젝트 환경

    | 분류     | 기술 스택/도구 | 버전     |
    | -------- | -------------- | -------- |
    | 언어     | Python         | 3.7.6    |
    | 머신러닝 | Numpy          | 1.18.1   |
    |          | Scipy          | 1.4.1    |
    |          | Scikit-learn   | 0.22.1   |
    | 딥러닝   | Tensorflow     | 2.0.0    |
    |          | Keras          | 2.2.4-tf |
    | 시각화   | Matplotlib     | 3.1.3    |
    |          | Pillow         | 7.0.0    |
    | 기타     | Anaconda       | 4.8.2    |





## 🚀 Quick Start

- 주어진 데이터셋을 전처리하고 모델을 학습시킵니다.

  ```bash
  $python train.py
  ```

  - 포함한 `preprocess.py` 의 전처리 함수들을 실행합니다.
    - `train_datasets` & `test_datasets` 를 불러오거나, 저장하여 불러옵니다.
    - `tokernizer` 를 불러옵니다. 
  - 모델 학습을 수행하며 손실을 출력합니다.

- 학습시킨 모델을 검증하고 테스트 합니다.

  ```bash
  $python predict.py
  ```

  - 테스트 데이터에서 임의의 이미지를 뽑아 캡션을 생성합니다.
  - 이미지와 캡션을 시각화하고 실제 캡션을 함께 출력하여 비교할 수 있게 합니다.

- 정규화한 이미지를 보여줍니다.

  ```bash
  $python doc/image_normalization_test.py
  ```

  - 이미지를 어떤 값으로 정규화 할 것인지 결정하기 위해 총 다섯가지의 방법을 사용하여 정규화한 이미지를 모두 띄워줍니다. 이를 각각 비교하고 어떠한 값을 활용할 지 결정할 수 있습니다.

    1. 오리지널 
    2. 넘파이로 min-max  값을 가져옵니다.
    3. 넘파이로 mean-std  값을 가져옵니다.
    4. 텐서플로우로 mean-var 값을 가져옵니다.
    5. 텐서플로우에서 정규화시켜주는 방법

- 텍스트 데이터 전처리

  ```bash
  $python doc/tokenizer_sample.py
  ```

  - datasets/ 아래에 tokenizer_sample.pkl 파일이 저장됩니다.
  - 저장 된 tokenizer 를 불러와 주어진 caption 을 토큰화 하고, sample 로 두 개 출력합니다.





## ⚡ Features

### 1. 프로젝트 구조

```
.
├── product
|   ├── index.js
|   ├── product.js
|   └── product.test.js
├── user
|   ├── index.js
|   ├── user.js
|   └── user.test.js
```



### 2. 데이터 ...전처리........꾸에엥ㄱ

#### 데이터셋 분리 및 저장

- `train.py` 에서 `train_datasets_path` 와 `test_datasets_path` 를 불러올 때, 해당 파일이 생성되어 있는지 여부를 판단합니다. 아직 데이터셋이 없다면 전체 데이터셋을 분리, 저장하는 `dataset_split_save()` 함수를 호출합니다.

  ``````python
  # train.py
  
  train_datasets_path = os.path.join(BASE_DIR, 'train_datasets.npy')
  test_datasets_path = os.path.join(BASE_DIR, 'test_datasets.npy')
  if not os.path.exists(train_datasets_path):
      # 이미지 경로 및 캡션 불러오기
      dataset = preprocess.get_path_caption(config.caption_file_path)
      preprocess.dataset_split_save(dataset, BASE_DIR, config.test_size)
      print('dataset 을 train_datasets 과 test_datasets 으로 나눕니다.')
  else:
      print('저장 된 train_datasets 과 test_datasets 을 사용합니다.')
  ``````

- `test-size` 값을 config 로 설정하여 train-test 비율을 지정할 수 있습니다. 

#### 데이터 파일 로드

- `config` 설정으로 train/test 중 어느 데이터셋을 가져올지 지정합니다.

  샘플링 여부를 설정합니다.

- 텍스트 데이터는 `<start>` `<end>` 토큰을 추가하여 반환합니다.

#### 텍스트 토큰화



#### 이미지 정규화

- 이미지를 불러올 때 함께 수행합니다.

  ``````python
  def load_image(image_path):
      img = tf.io.read_file(image_path)
      img = tf.image.decode_jpeg(img, channels=3)
      img = tf.image.resize(img, (299, 299))
      img = tf.keras.applications.inception_v3.preprocess_input(img)
      return img, image_path
  ``````

  InceptionV3 모델을 사용하기 위해, 해당 모델에 적합한 형태로 전처리 하는 과정을 추가하여 이미지를 반환합니다.

#### 이미지 증강

``````python
from imgaug import augmenters as iaa
# imgaug 모듈을 불러옵니다
``````

파이썬 라이브러리 `imgaug` 사용하여 Image Augmentation(이하 이미지 증강) 을 수행합니다.

Sequentianl 안에서 여러 종류의 이미지 증강을 수행합니다.

BlendAlpha 로 알파블렌딩 작업을 추가해 이미지를 증강할 수 있습니다.

- 이미지 증강 작업은 `feature_extraction.py` 에서 호출합니다.

  



### 3. 실..행....



### 4. 결과물













## 🕵️‍♀️🕵️‍♂️ Documentation

### 프로젝트 관리

- [프로젝트 관리]()



### 멋진 팀원들이 정리한 멋진 자료들

- 솔지, [데이터 정규화와  CNN, 손실함수]()
- 솔지, [numpy 배열 다루기]()
- 수민, [프로젝트 기본 개념과 이미지 캡셔닝 프로젝트 아키텍처]()
- 찬우, [퍼셉트론 Perceptron]()
- 찬우, [머신러닝 기본 개념]()
- 찬우, [CNN]()
- 수진, [데이터 전처리 - 이미지 특성 추출]()
- 수진, [학습 모델]()
- 수진, [Linear_Regression]()

- 현동, [Tokenizer]()






## 💖 Team

> SSAFY 2기 4반 5팀 : 김수민, 양찬우, 이수진, 조현동, 최솔지 

**너는**  ![team](C:\Users\multicampus\Desktop\Project\s02p22a405\doc\images\team.jpg)







------

여러분이 사랑하는 자존감이 낮은 친구에게 [이 영상](https://youtu.be/d4XGFYNcUEc)을 보여주세요. 💕