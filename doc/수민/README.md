# Sub PJT II (Week 2, 3) 이미지 캡셔닝 활용 시스템

이미지 캡셔닝 구현

## 1. 프로젝트 개요

### Task

이미지 담당 / 텍스트 담당 - 이미지 캡셔닝 모델 구현

이미지 캡셔닝이란? 이미지를 묘사하는 문장을 생성하는 것

1. 이미지가 입력으로 들어온다

2. 이미지 모델로 특성 뽑는다

3. 텍스트 모델에 전달한다

4. 이미지를 묘사하는 텍스트를 생성한다

### Content

- CNN - 이미지 데이터에 적합한 Convolution 신경망
  - 이미지에서 물체의 형태를 인지하거나 색깔을 구별하는 등 "특성"을 뽑아내는 데 사용된다
- RNN - 순서가 있는 데이터에 적합한 순환 신경망
  - 뽑아낸 특성을 바탕으로 문장을 생성한다
- 이 두 모델을 합쳐 Image Captioning 모델 구현

## 2. 프로젝트 목표

1. 컨볼루션 인공 신경망(CNN) 이해

2. 순환 신경망(RNN) 이해

3. 이미지 캡셔닝 모델 이해

4. 데이터셋 분할과 성능 최적화 이해

5. 팀별 서비스 기획, 데이터셋 검색 및 모델 선정

## 3. 필수 지식 학습

### 자연어 처리

#### NLP

##### 토큰화

[딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/21698)

- **말뭉치 또는 코퍼스(corpus)**

  - 자연 언어 연구를 위해 특정한 목적을 가지고 언어의 표본을 추출한 집합 / 조사나 연구 목적에 의해서 특정 도메인으로부터 수집된 텍스트 집합

  - 코퍼스 데이터를 토큰화(tokenization) & 정제(cleaning) & 정규화(normalization)

- **토큰화(tokenization)**

  - 토큰(token)이라 불리는 단위로 나누는 작업을

  - 갖고 있는 코퍼스가 정제되지 않은 상태라면, 코퍼스는 문장 단위로 구분되어있지 않을 가능성이 큼

  - 이를 사용하고자 하는 용도에 맞게 하기 위해서는 문장 토큰화가 필요할 수 있음

- **품사 태깅(part-of-speech tagging)**

  - 단어 토큰화 과정에서 각 단어가 어떤 품사로 쓰였는지를 구분해놓는 것

##### 정수 인코딩

케라스의 텍스트 전처리

[딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/31766)

- 각 단어를 고유한 정수에 맵핑

- 보통 단어에 대한 빈도수 기준으로 정렬한 뒤 부여

- 단어를 빈도수 순으로 정렬한 vocabulary 만들고

- 빈도수가 높은 순서대로 낮은 숫자부터 정수를 부여하는 방법

##### 원 핫 인코딩

- 단어 집합: 텍스트의 모든 단어의 중복을 허용하지 않고 모아 놓음

- 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식

##### 임베딩

[keras_practice](keras_practice.ipynb)

정수 인코딩 된 벡터를 임베딩하는 방법

https://subinium.github.io/Keras-6-1/

#### RNN

##### 순환 신경망 기초

순환 신경망에 대한 기본적인 설명과 예시 코드

https://excelsior-cjh.tistory.com/183?category=940400

##### LSTM & GRU

기본 RNN 을 개선한 LSTM 과 GRU 에 대한 설명

https://datascienceschool.net/view-notebook/770f59f6f7cc40c8b6dc98dddd06c6c5/

### 이미지 캡셔닝 프로젝트 아키텍처

#### 1. 기본 아키텍처

1. 이미지를 입력으로 받아 CNN 모델을 통해 이미지의 특징을 추출한다.

2. 추출한 특징은 RNN 모델로 전달된다.

3. RNN 모델은 여기에 추가로 토큰화된 캡션 데이터 일부를 입력받는다.

   (RNN 모델은 CNN모델을 통해 추출한 이미지의 특성과 캡션 벡터 2가지를 입력으로 받아 학습한다)

#### 2. 프로젝트 구조

##### 1) 전처리 과정 (= Req 1 ~ Req 2)

- 전처리 후 train 혹은 test 용 데이터셋을 저장함

- tokenize 에 사용한 tokenizer 객체를 pickle 에 저장

![req2](req2.jpg)

###### 한 번만 시행해도 되는 전처리

- 캡션 데이터를 토큰화한다.

  - csv 파일을 토대로 실제 이미지 경로와 "이미지에 해당하는 토큰화된 캡션"을 묶는다.

- 전체 데이터셋을 분할한다.

  - 학습용, 테스트용 데이터로 분할해서 저장해 놓는다.

  - 어떤 단어가 토큰에 해당하는지 맵핑된 정보를 기록해놓은 토크나이저 또한 저장한다.

  - pre-trained?

###### 매번 진행하는 전처리

- 학습하는지 또는 테스트를 진행하는지 따라 데이터셋(이미지 경로와 토큰화된 캡션 있음) 다른 것 불러오고 // 토크나이저도 불러온다.

- 데이터셋에는... 이미지 경로와 토큰화된 캡션 있음... 실제 이미지 데이터와 토큰화된 캡션을 바인딩해서 텐서플로우 데이터셋으로 만든다.

- 텐서플로우 데이터셋을 만드는 과정에서 / 데이터 랜덤성 추가를 위해 데이터의 순서를 바꾸기도 하고 이미지의 경우 뒤집거나... 한다.

- 단, 훈련할 때만 적용한다는 뜻인듯? 테스트시에는 순서 바꾸기, Aug 안 들어가게 구현한다.

##### 2) 학습 과정

- `tf.data.Dataset` 에는 Encoder 의 입력이 되는 이미지 데이터 또는 미리 뽑힌 특성 벡터가 들어 있음. 토큰화된 캡션도 쌍으로 들어 있음.

- 이미지 데이터 또는 미리 뽑힌 특성 벡터는 Enocder에 들어가 Decoder 의 입력 형식에 맞게 변환되어 나온다.

- Encoder 의 결과값은 Decoder 로 전달된다. 이와 동시에 `<start>` 토큰의 인덱스가 Decoder 로 전달된다.

- Decoder 는 순차적으로 이미지를 묘사하는 문장의 단어들을 생성한다.

- 이 때 단어들은 토큰의 가지 수만큼 길이를 가지는 벡터로 나온다.

- 이 결과값을 정답 캡션과 비교해 손실을 계산한다

- 이 손실을 기반으로 모델의 변수들을 학습하여 최적화를 진행한다.

##### 3) 테스트 과정

---

### 모두를 위한 딥러닝 입문 시즌2 - Tensorflow

[Youtube](https://www.youtube.com/playlist?list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C)

[Github](https://deeplearningzerotoall.github.io/season2/lec_tensorflow.html)

#### PART 2: Basic Deep Learning

##### Lec 08-1: 딥러닝의 기본 개념: 시작과 XOR 문제

- 활성화 함수(Activation function)

- XOR 문제

  - XOR 을 Linearly 분리 불가능

- 퍼셉트론(Perceptrons)

##### Lec 08-2: 딥러닝의 기본 개념2: Back-propagation 과 2006/2007 '딥'의 출현

- 오차 역전파(Backpropagation)

- 가중치 초기화(Weight initialization)

- CIFAR, ImageNet

##### Lec 09-1: XOR 문제 딥러닝으로 풀기

#### PART 4: Recurrent Neural Network

Lec 12: NN의 꽃 RNN 이야기

Lab 12-0: rnn basics

Lab 12-1: many to one (word sentiment classification)

Lab 12-2: many to one stacked (sentence classification, stacked)

Lab 12-3: many to many (simple pos-tagger training)
