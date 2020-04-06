# README
![](https://img.shields.io/badge/version-1.0.0-green.svg) ![](https://img.shields.io/badge/created__at-20.04.03-yellow.svg) ![](https://img.shields.io/badge/updated__at-20.04.03-blue.svg)

- 특화프로젝트 "이미지 캡셔닝 활용 시스템"  Sub PJT 2 : 이미지 캡셔닝 기능 구현

- SSAFY 2기 4반 5팀 : 김수민, 양찬우, 이수진, 조현동, 최솔지 



## 🔊 목표

- Sub2 PRJ 스프린트 #2  `ver.1.0.0`
  - 컨볼루션 신경망과 순환 신경망을 이해하고 명세에 따라 기능을 구현합니다.
  - 프로젝트 환경을 구성하고 공유합니다.
  
    

## 💻 구현 내용

1. 이미지 데이터 전처리
   1. 이미지 파일 로드
   2. 이미지 정규화
2. 텍스트 데이터 전처리
   1. Tokenizer 저장 및 불러오기
   2. 텍스트 데이터 토큰화 

3. Dataset 관리
   1. train_dataset.npy && test_dataset.npy 
      - captions.csv 파일을 7:3 비율로 **섞지않고**(이미지 1개+캡션 5개 고정) 나누어서 저장해두었습니다. 
      - 자료 구조를 확인할 수 있게 csv 파일도 함께 만들어두었습니다.
      - 이에 따라 아래 Quick Start 는 train_datasets 파일을 불러오고, 사용하여 결과를 보입니다.



## 🚀 Quick Start

- 가상환경 설정

  ```
  conda env create -f AI.yaml
  ```




- 이미지 데이터 전처리

  ```
  python doc/image_normalization_test.py
  ```

  - 정규화를 한 이미지를 보여줍니다.

  - 이미지를 어떤 값으로 정규화 할 것인지 결정하기 위해 총 다섯가지의 방법을 사용하여 정규화한 이미지를 모두 띄워줍니다. 이를 각각 비교하고 어떠한 값을 활용할 지 결정할 수 있습니다.

    1. 오리지널 
    2. min max  
    3. mean std 
    4. mean var (텐서플로우)  
    5. 텐서플로우에서 정규화시켜주는 방법

  - 자세한 내용은 doc/이미지정규화.md 파일을 참고해 주세요.

    

- 텍스트 데이터 전처리

  ```
  python doc/tokenizer_sample.py
  ```

  - datasets/ 아래에 tokenizer_sample.pkl 파일이 저장됩니다.
  - 저장 된 tokenizer 를 불러와 주어진 caption 을 토큰화 하고, sample 로 두 개 출력합니다.
  - 자세한 내용은 doc/텍스트전처리_토큰화.md 파일을 참고해 주세요.





## 💥 Notice

- train.py, config.py 에 구현 된 내용은 Sub01 과 동일한 결과물이며, 정상적으로 기능하나, 위의 Quick Start 만 실행하시기를 권장드립니다.