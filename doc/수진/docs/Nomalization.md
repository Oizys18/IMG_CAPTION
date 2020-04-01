### Preprocessing

#### Nomalization

이미지 정규화를 하는 이유? [참고](https://stackoverflow.com/questions/4674623/why-do-we-have-to-normalize-the-input-for-an-artificial-neural-network) 

1. 수렴속도가 빨라짐.

2. local optimum에 빠지는 것을 방지.

3. 모든 데이터가 동일한정도의 스케일로 반영 되도록 해줌.

   

크게 두가지 방법이 있다. (물론 더 많다..)

- Min-Max Normalization 
  $$
  X = {x -x_{min}\over x_{max}-x_{min}}
  $$
  
- Standardization (Z-score Normalization) 
  $$
  z = { x - \mu \over \sigma} \\
  {\mu} : Mean, \; {\sigma} :Standard \; Deviation
  $$
  

직접 수식을 구현 해도 되지만 많은 라이브러리에서 스케일링 함수를 지원 한다 : [sckit-learn](https://scikit-learn.org/stable/modules/preprocessing.html/)

- `scale` 함수는 zero mean, unit variance를 가지는 분포로 데이터를 scaling 해준다. (axis=0 이 default)

- `StandardScaler`

  한장 씩 하는건가?  [How To Prepare Your Data For Machine Learning in Python with Scikit-Learn](https://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/)
  
  여러장 씩 하는건가? 
  
  https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
  
  -> 이 글을 보면 채널 별로 하는듯 : 아무튼 이정도만 해보고 모델 다 짜고 다시 찾아봐도 될 듯

#### Augmentation 

모델 학습 성능 향상을 위함. 다양한 방법이 있다 [참고](https://deepestdocs.readthedocs.io/en/latest/003_image_processing/0030/)

- Crop 
- Rotate 
- Flip (보통 좌우 반전을 많이 함)
- Translate 
- Resize 
- Zero filling , Nearest neighbot, Rolling  
- Zooming  
- RGB 랜덤 노이즈, (PCA로 중요성붙을 찾아 랜덤하게 노이즈를 더할 수 도 있다.)
- Noise Addition (가우시안 랜덤 노이즈)

---

참고 

[모두를 위한 딥러닝 7-1](https://www.youtube.com/watch?v=oSJfejG2C3w&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=20)

 [데이터 표준화와 정규화의 차이점](https://soo-jjeong.tistory.com/123?category=874990)

[패션 MNIST 데이터 분류](https://www.tensorflow.org/tutorials/keras/classification?hl=ko)

[scikit-learn의 전처리 기능](https://datascienceschool.net/view-notebook/f43be7d6515b48c0beb909826993c856/) 