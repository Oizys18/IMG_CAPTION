# 이미지 캡셔닝 기능 구현

![](https://img.shields.io/badge/version-2.0.0-green.svg) ![](https://img.shields.io/badge/created__at-20.04.03-yellow.svg) ![](https://img.shields.io/badge/updated__at-20.04.10-blue.svg)

> 인공지능 프로젝트 "이미지 캡셔닝 활용 시스템" Sub PJT 2



-----

## Table of Contents

- [Installation](#installation)
- [Quick Start](#Quick Start)
- [Features](#features)
- [Documentation](#Documentation)
- [Team](#Team)

---------





## Installation 

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






## Quick Start

- 주어진 데이터셋을 전처리하고 모델을 학습시킵니다.

  ```bash
  $python train.py
  ```

- 학습시킨 모델을 검증하고 테스트 합니다.

  ```bash
  $python predict.py
  ```

  - ㅇㅇㅇ
  
    

- 정규화한 이미지를 보여줍니다.

  ```bash
  $python doc/image_normalization_test.py
  ```

  - 이미지를 어떤 값으로 정규화 할 것인지 결정하기 위해 총 다섯가지의 방법을 사용하여 정규화한 이미지를 모두 띄워줍니다. 이를 각각 비교하고 어떠한 값을 활용할 지 결정할 수 있습니다.

    1. 오리지널 
  2. min max  
    3. mean std 
    4. mean var (텐서플로우)  
    5. 텐서플로우에서 정규화시켜주는 방법
  
    

- 텍스트 데이터 전처리

  ```bash
  $python doc/tokenizer_sample.py
  ```

  - datasets/ 아래에 tokenizer_sample.pkl 파일이 저장됩니다.
  - 저장 된 tokenizer 를 불러와 주어진 caption 을 토큰화 하고, sample 로 두 개 출력합니다.





## Features

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



### 3. 실..행....



### 4. 결과물













## Documentation

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






## Team

> SSAFY 2기 4반 5팀 : 김수민, 양찬우, 이수진, 조현동, 최솔지 

**너는**  ![team](C:\Users\multicampus\Desktop\Project\s02p22a405\doc\images\team.jpg)







------

여러분이 사랑하는 자존감이 낮은 친구에게 [이 영상](https://youtu.be/d4XGFYNcUEc)을 보여주세요. 💕