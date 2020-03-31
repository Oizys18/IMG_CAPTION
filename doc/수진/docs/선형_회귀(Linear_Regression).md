# 선형 회귀 : Linear Regression



✔ 선형적 (Linear)

어떤 function, operation, System이 '선형적' 이라고 말하려면 아래 두 가지 조건을 만족해야 한다.

(Superposition principle을 만족해야한다.)

1. Additivity (첨가성) 
   $$
   f(x_1+x_2) = f(x_1) + f(x_2)
   $$
   
2. Homogeneity ( 균질성 )
   $$
   f(ax_1) = af(x_1)
   $$



대표적으로 미분, 적분, Matrix 연산이 선형적인 operaion이다.

선형적인 문제는 특정 input에 대한 output을 예측할 수 있어 비 선형 문제에 비해 간단히 해답을 찾을 수 있다. 



✔ 회귀(regression)

머신러닝의 지도학습은 크게 회귀(regression)와 분류(classification)로 나뉜다.

회귀 분석은 연속적인 데이터를 예측할 때 사용된다.

사람의 몸무게별 키, 아파트 면적 별 가격 



## 선형 회귀 

선형 회귀는 가장 간단한 회귀 모델임.

학습 과정은 크게 가설을 세우고  error를 구해서, 새로운 가설을 세우는 과정(1 epoch이라 함)을 반복한다.

### 1. 가설 세우기

기본식은 다음과 같다.
$$
H(W, b) = Wx + b
$$

`H:Hypotheses, W:Weight, b:bias `

여기서 x, y 가 상수 $W, b$ 가 우리가 구해야 할 값이다.

맨 처음 가설은 (W, b) 값을 임의로 넣어주고 구한다.



### 2. Error 구하기

이제 실제 값과 우리가 세운 가설의 오차를 계산한다.

오류 지표는 MSE(Mean Squared error; 평균 제곱 오차)를 사용한다. 에러를 제곱하면,  `Gradient Descent` 계산 시 편미분에 용이하고 오류가 작아도 쉽게 비교할 수 있다.

 만약 아래와 같다면, 

$$
y_i^* = 실제 \; 데이터의 \; y값  \\
y_i = H(W,b) \;\\
$$
MSE는 다음과 같이 나타낼 수 있다.
$$
MSE =\frac{1}{n} \Sigma(y_i - y_i^*)^2
$$

선형 회귀에서 오류 함수는 (Cost function, Loss function 이라고도 함) Convex 하다.

따라서, 미분 (우리가 찾고자 하는 W, b에 대해 각각 편미분)시 0이 되거나, 가까워지는 점이 있다.

이 점에서 W, b가 바로 우리가 찾고자 하는 모델이다.



### 3. 새로운 가설 세우기

위에서 구한 오류를 바탕으로 새로운 가설을 세운다.

만약 오류가(=기울기가) 음수라면  W, b 값에 *일정한 값을 더해주고 양수라면 W, b 값에 *일정한 값을 빼준다. 이러한 알고리즘을 **Gradient Descent** 라고 한다.

*일정한 값 =  W(혹은 b)의 기울기
$$
W = W -\alpha{\partial{cost}(W,b)\over\partial W}
$$


learning rate ($\alpha$): 학습률 (초기에 직접 설정)

위 1 ~ 3 과정으로 W, b의 값을 찾는 것을 train 이라 함.



----

마지막이라는 생각으로 계산을 해본다.



----

참고

[선형이 의미하는 것](https://brunch.co.kr/@gimmesilver/18)

[머신러닝의 기초- 선형 회귀](https://www.youtube.com/watch?v=ve6gtpZV83E)

[square error를 쓰는 이유](https://nittaku.tistory.com/284)

[딥러닝을 이용한 자연어처리 입문](https://wikidocs.net/21670)

[학습률](https://bioinformaticsandme.tistory.com/130)

[Tex_문법](https://ko.wikipedia.org/wiki/위키백과:TeX_문법)