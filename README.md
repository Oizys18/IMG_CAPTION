# README

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

  

- 텍스트 데이터 전처리

  ```
  python doc/tokenizer_sample.py
  ```

  - datasets/ 아래에 tokenizer_sample.pkl 파일이 저장됩니다.
  - 저장 된 tokenizer 를 불러와 주어진 caption 을 토큰화 하고, sample 로 두 개 출력합니다.





## 💥 Notice

- train.py, config.py 에 구현 된 내용은 Sub01 과 동일한 결과물이며, 정상적으로 기능하나, 위의 Quick Start 만 실행하시기를 권장드립니다.