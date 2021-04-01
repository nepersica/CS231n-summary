# Lecture 3 (Loss Functions and Optimization)
- 학습일: 2021/04/01
- 주제: 손실함수와 최적화


![image](https://user-images.githubusercontent.com/37270069/113320077-13ddcd00-934d-11eb-8ff3-ab73cdd14d12.png)

앞서 2장에서는 linear classifier에 W값을 사용한다는 것을 배웠다.
위 사진에서 고양이, 자동차, 개구리의 각 class에 대한 score를 보면,
자동차는 automobile이 가장 높은 score로 나타났기에 잘 분류했다고 볼 수 있지만,
개구리는 frog가 굉장히 낮은 score를 나타내었기에 잘 분류하지 못하였다고 볼 수 있고,
즉, 이러한 좋지 못한 score가 나타나도록 영향을 준 W값이 좋지 못하다는 것을 의미한다.
이는 좋은 score 값을 가지기 위해서는 좋은 W값을 가져야 한다는 것이고,
따라서 어떤 W값이 가장 좋은지를 결정하기 위해서는 W값이 좋은지 나쁜지에 대해 정량화할 방법이 필요하다.
이것이 바로 Loss Function, 즉 '손실함수' 이다.
이러한 손실함수를 통해 W값이 좋은지 나쁜지를 알게 되었다면, 우리가 실제로 원하는 것은 W가 될 수 있는 모든 경우의 수에 대해서 가장 좋은 W가 무엇인지를 찾는 것이다.
이 과정을 Optimization, 즉 '최적화' 라고 한다.
이번 3장에서는 이 Loss Function, Optimization에 대해 핵심적으로 알아본다.


## Loss Functions
### 기본적인 Loss

![image](https://user-images.githubusercontent.com/37270069/113320157-26f09d00-934d-11eb-994b-b1e593b358a3.png)

Loss는 모델에 의한 예측 이미지 label과 실제 이미지 label의 일치하지 않는 정도를 말한다.
일반적인 Loss를 구하는 방법은, 각 클래스 별 loss를 각각 구하고, 이를 다 더한 후, 마지막으로 전체 개수로 나누어 평균값을 만드는 것이다.
위 그림의 수식으로 다시 설명하자면, f(x_i,W)는 모델에 의한 예측 이미지 label이며, y_i는 실제 이미지 label이다.
예측 이미지 label과 실제 이미지 label을 통해 L_i(f(x_i,W,y_i)라는 Loss를 구하고, 이러한 Loss를 모두 합쳐 데이터의 개수로 나누면 전체 데이터 셋에 대한 Loss를 구할 수 있다.
이러한 기본적인 Loss에 대한 이해를 바탕으로 이번 강의에서는 SVM Loss와 Softmax Classifier에 대해 알아본다.


### SVM Loss

![image](https://user-images.githubusercontent.com/37270069/113321223-58b63380-934e-11eb-9648-dea8757e4b58.png)

첫 번째로 알아볼 Loss는 서포트 벡터 머신(SVM) Loss 이다.
SVM Loss는 각각의 트레이닝 데이터에서의 Loss인 L_i를 구하기 위해 다음과 같은 방법을 사용한다.
각 이미지의 실제 label에 해당하는 score를 s_yi라 하고, 이 외의 나머지 label에 해당하는 score를 s_j라고 할 때,
정답 클래스의 score인 s_yi가 오답 클래스의 score인 s_j보다 높고, 그 격차가 일정 마진(safety margin) 이상이라면 Loss는 0이다.
만약 위의 조건에 맞지 않는다면, (s_j - s_yi + safety margin)의 값을 Loss로 한다.
이를 수식으로 간단하게 나타내면, max(0, s_j - s_yi + 1)이 된다.
따라서, 정답 클래스의 score가 safety margin을 고려하더라도 오답 클래스의 score보다 높으면 Loss는 0이 되는 것이며, Loss가 0이라는 것은 매우 잘 예측했다는 것을 의미한다.
만일 Loss가 0이 아닌 다른 값 k가 나왔다고 가정하면, 이는 분류기가 k만큼 해당 training set을 잘못 분류하고 있다는 뜻이며, 이러한 k가 바로 우리가 목표로 하는 '정량적 지표'가 되는 것이다.

![image](https://user-images.githubusercontent.com/37270069/113321153-463bfa00-934e-11eb-9f06-ae854f2b15d1.png)

즉, 0과 다른 값의 최댓값, Max(0, value)과 같은 식으로 손실 함수를 만드는데,
이를 그래프로 나타내면 위 그래프처럼 나오게 된다.
이 모양이 경첩처럼 생겼다고 해서 이런 류의 손실함수를 "hinge loss"라고 부르기도 한다.
그래프에 관해서 설명을 붙이자면, x축은 정답 클래스의 score인 s_yi이고, y축은 Loss이다.
x축에 해당하는 정답 클래스의 score 값이 높아질수록 y축에 해당하는 Loss가 선형적으로 줄어드는 것을 볼 수 있다.

그렇다면 SVM Loss에서 사용하는 safety margin(위 사진에서는 safety margin=1)은 어떻게 결정되는 것일까?
safety margin은 손실 함수에서 상수 term으로 자리잡고 있어서 보기 불편하거나 의문을 가질 수 있는데,
SVM Loss가 중요시하는 것은 각 score가 정확히 몇인지가 아니라, 여러 score 간의 상대적인 차이이다.
다시 말해, SVM Loss가 관심 있어 하는 것은 오로지 정답 score가 다른 오답 score 보다 얼마나 더 큰 score를 가졌는지 이다.
따라서 행렬 W를 전체적으로 스케일링한다고 생각해보면, 결과 score도 이에 따라 스케일이 바뀔 것이고, safety margin에 해당하는 1이라는 parameter값은 없어지고 W의 스케일에 의해 상쇄된다는 것을 알 수 있다.

SVM Loss에 대한 직관적인 이해를 위해 몇 가지 질문들로 다시 접근해본다.

1) 만약 정답 score값이 오답 score값보다 훨씬 클 때, 정답 score값이 조금 변한다면 Loss에도 변화가 있을까?
결론부터 말하면, Loss는 변하지 않는다.
정답 클래스의 score값이 오답 클래스의 score값보다 훨씬 크기에, score값이 조금 바뀐다고 해도 서로 간의 간격은 여전히 유지될 것이고, 결국 Loss는 변하지 않고 계속 0이라는 것이다.
이는 SVM Loss의 근본적인 특성 중 하나이다.
SVM Loss는 score값이 정확히 몇인지는 중요하지 않고, 정답 클래스의 score값이 오답 클래스의 score값보다 높은지만 관심 있기 때문에, 데이터에 '둔감'하다는 것이다.

2) SVM Loss의 최솟값과 최댓값은 얼마일까?
모든 클래스의 걸쳐서 정답 클래스의 score가 가장 크다면, 해당 training set의 최종 loss는 0이다.
만약 정답 클래스의 score가 엄청 낮은 음수 값을 가지고 있다면, loss는 무한대가 될 것이다.
따라서 SVM Loss의 최솟값은 0, 최댓값은 무한대이며, 이는 위의 SVM Loss 그래프를 통해 쉽게 알 수 있다.

3) 파라미터를 초기화하고 처음부터 학습시킬 때 보통 W를 임의의 작은 수로 초기화하는데, 그렇다면 처음 학습 시에는 결과 score가 임의의 일정한 값을 갖게 된다. 이때, 만약 모든 score가 거의 '0에 가깝고', '값이 서로 거의 비슷하다면' Loss는 어떻게 될까? (safety margin = 1)
정답은 'Class 수 - 1'이다.
SVM Loss는 loss를 계산할 때 정답이 아닌 클래스를 순회한다.
즉, (Class 수 - 1)개의 클래스를 순회하는데, 비교하는 두 score가 거의 비슷하기에, safety margin에 해당하는 값을 얻게 된다. # max(0, 0 - 0 + (safety margin)) = safety margin
safety margin이 1이라고 하면, 최종 loss 값은 (Class 수 - 1) * (safety margin) = Class 수 -1 이 된다.
이는 디버깅 전략으로 굉장히 유용하다.
예를 들어, Training을 처음 시작할 때 Loss가 C-1이 아니라면 오류가 있다는 것을 의미하며, 이를 'sanity check'라고 부른다.

4) SVM Loss에서는 정답 클래스의 score를 빼고 계산을 하는데, 만약 정답 클래스의 score도 포함해서 계산을 하면 어떻게 될까? (safety margin = 1)
정답은 최종 Loss 값이 1 증가한다.
일반적으로 Loss가 0이 되어야 가장 좋다고 보는데, 정답 클래스의 score까지 포함하여 계산을 하게 되면 가장 좋은 Loss값이 1이 되고, 이는 보기 어색하다고 느껴진다.
더불어, 정답 클래스 score까지 포함하여 계산한다고 분류기가 학습이 더 잘되는 것이 아니기 때문에, SVM Loss는 정답 클래스의 score를 빼고 계산하는 것이다.

5) 최종 Loss를 전체 Loss의 합이 아닌, 전체 Loss의 평균으로 한다면 어떻게 될까?
정답은 아무 영향을 미치지 않는다.
SVM Loss는 각 score 값이 몇인지는 신경 쓰지 않기에, 전체 클래스의 수는 어차피 정해져 있으므로 평균을 취한다는 것은 그저 손실 함수를 re-scale 할 뿐이다.
따라서 sclae만 변할 뿐, 결과에는 영향을 미치지 않는다.

6) Loss Function을 제곱 항으로 바꾸면 어떻게 될까? 즉, max(0, s_j - s_yi + 1)이 아닌, pow(max(0, s_j - s_yi + 1), 2)로 한다면 어떻게 될까?
정답은 결과가 달라진다는 것이다.
제곱 항을 사용하게 되면 올바른 것과 잘못된 것 사이의 trade-off를 non-linear하게 바꿔주게 되는데, 이로 인해 손실 함수의 계산 자체가 바뀌게 된다.
이러한 방식의 Loss Function을 squared hinge loss라고 부르며, 실제로 손실 함수를 설계할 때 사용할 수 있는 한 가지 방법으로 종종 사용된다.
기본적인 SVM Loss에 해당하는 hinge loss는 앞서 말한 것처럼 score 값의 변화에 둔감하다.
다시 말해, '조금 잘못된 것'과 '많이 잘못된 것'을 크게 신경 쓰지 않는다는 것이다.
예를 들어, 많이 잘못된 것이 있다면 학습을 통해 Loss를 줄일 것인데, 그 줄어드는 Loss의 양이 '조금 잘못된 것'이던 '많이 잘못된 것'이던 큰 차이가 없다는 것이다.
하지만 제곱 항을 사용하는 squared hinge loss는 '많이 잘못된 것'의 Loss에 제곱을 하게 되면 '정말 많이 잘못된 것'가 되고, 이러한 잘못된 것에 대한 패널티가 급격하게 늘어나게 된다.
이처럼 직선이 아니라 non-linear한 곡선의 형태를 띄는 squared hinge loss는 '예측값이 매우 좋다' 혹은 '예측값이 매우 안 좋다' 등을 따지는 경우에 유용하게 사용될 수 있다.
따라서 어떤 Loss Function을 사용할 것이냐는 잘못된 것에 대해 얼마나 신경을 쓰고 있고, 그것을 어떻게 정량화 할 것인지에 달려있으며, 이러한 것은 손실 함수를 설계할 때 고려해야만 하는 문제이다.


## Regularization
우리의 목표는 결국 Loss가 0이 되게 하는 것이다. 그렇다면 과연 Loss가 0이 되게 하는 W값은 유일하게 하나만 존재하는 값일까?
정답은 Loss가 0이 되게 하는 W값은 여러 개 존재한다.
앞서 말한 것처럼 W의 scale은 변할 수 있으며, loss가 0이 되게 하는 W값에 n배를 한다 해도 loss는 여전히 0이다.
이는 굉장히 중요한 의미를 내포하고 있는데, 다양한 W값 중 해당 training 과정에서 loss가 0이 되는 W값을 선택하는 것은 모순적이라는 것이다.
우리가 실제로 목표로 하는 것은 training data를 통해 분류기를 학습시키고, 이를 통해 test data를 예측하는 것이다.
즉, training data에 완벽한 W값을 찾는 것이 아닌, test data에 높은 성능을 보이는 W값을 찾는 것이 목표이다.
W값은 unique하지 않기에, 만약 training data에 완벽한 W값을 찾았다고 해서 그 W값이 test data에 적합한 W값이 아닐 수 있다는 것이다.
이를 과적합(Over-Fitting)이라고 하며, 이러한 문제는 기계학습에서의 굉장히 중요한 문제이다.
그리고 보통 이러한 과적합을 해결하는 방법을 통틀어 'Regularization' 이라고 한다.

![image](https://user-images.githubusercontent.com/37270069/113320403-7040ec80-934d-11eb-93ce-62c9ec32f144.png)

Regularization은 보통 손실 함수에 'Regularization Term'을 추가하는데, 이는 모델이 조금 더 '단순'한 W를 선택하도록 도와준다.
여기서 '단순'하다는 것은 '일반적'이라는 의미로 해석할 수 있는데, 앞으로 일어날 새로운 일(test data)에 대해 적합(올바르게 예측)할 가능성이 높음을 의미한다.
따라서 'Data Loss Term'이 training data에 fit하게 하려할 때, 'Regularization Term'을 통해 'Regularization Penalty'를 주면서 최적의 W값을 찾아 나가는 것이다.
추가로 'Regularization Term'에 붙어있는 hyper parameter lambda는 모델을 훈련시킬 때 고려해야 할 중요한 요소 중 하나로, Regularization의 강도 값이다.
Regularization 방법으로는 L1 regularization, L2 regularization 등이 있으며, 더불어 dropout, batch normalization 등도 사용된다.


### Softmax Classifier(Multinomial Logistic Regression)
앞서 설명한 SVM Loss의 경우 정답 클래스 score값이 오답 클래스 score값보다 높은지에만 관심이 있었고, 실제 score 값은 중요하지 않았다.
하지만 Multimomial Logistic Regression의 손실 함수는 score 값 자체에 추가적인 의미를 부여한다.

![image](https://user-images.githubusercontent.com/37270069/113320466-8189f900-934d-11eb-8ed0-7a5be0d2dec1.png)

Softmax Function은 모든 score 값에 지수를 취해서 양수가 되게 만들고, 그 지수들의 합으로 다시 정규화 시킨다.
이를 통해 Softmax Function을 거치면 결국 확률 분포를 얻을 수 있게 되고, 그것은 바로 해당 클래스일 확률이 되는 것이다.
이러한 확률은 0과 1 사이의 값이며, 모든 확률들의 합은 1이 된다.
결국 우리가 원하는 것은 정답 클래스에 해당하는 클래스의 확률이 1에 가깝게 계산되는 것이고, 따라서 Loss는 '-log(정답 클래스 확률)'이 된다.
'-log'를 취하는 이유는 log는 단조 증가 함수이며, 확률 값을 최대화 시키는 것보다 log를 최대화 시키는 것이 쉽다.
또한 손실 함수는 '얼마나 좋은지'가 아니라 '얼마나 나쁜지'를 측정하는 것이기에 log에 마이너스를 붙인다.
예를 들어, 정답 클래스 확률이 1이면 'Loss = -log(정답 클래스 확률) = -log(1) = 0'이므로 Loss가 0이고, 정답 클래스 확률이 0이면 같은 'Loss = -log(0)'이므로 Loss가 무한대가 됨을 알 수 있다.

이번에도 역시 Softmax Loss에 대한 직관적인 이해를 위해 몇 가지 질문들로 다시 접근해본다.

1) Softmax Loss의 최솟값과 최댓값을 얼마일까?
정답은 최솟값은 0이고, 최댓값은 무한대이다.
이는 앞서 예를 든 것처럼, 정답 클래스 확률이 1이면 Loss가 0이 되고, 정답 클래스 확률이 0이면 Loss가 무한대가 된다.
물론 이론상으로는 위와 같지만, 컴퓨터는 무한대 계산을 하지 못하고 유한 정밀도 때문에 실제로 최댓값(무한대)과 최솟값(0)에 도달할 수는 없다.

2) 만약 score 값이 모두 0 근처에 모여있는 작은 수일 때, Loss는 어떻게 될까?
정답은 -log(1 / 클래스 개수) 이다.
이는 위의 SVM Loss의 sanity check와 마찬가지로 첫 번째 iteration에서 사용해볼 만한 유용한 디버깅 전략이다.

SVM Loss와 Softmax Loss의 차이를 예를 들어 비교해보면 다음과 같다.

![image](https://user-images.githubusercontent.com/37270069/113320606-a54d3f00-934d-11eb-9a06-bb15022b1c6f.png)

SVM Loss에서는 정답 클래스 score 값과 오답 클래스 score 값 간의 margin을 통해 Loss를 계산한다.
반면, Softmax Loss에서는 각 클래스 확률을 통해 Loss를 계산한다.
또한 SVM Loss가 score 값의 미미한 변화에 둔감하여 일정 margin을 넘기기만 하면 더 이상 성능 개선을 신경 쓰지 않는 반면, Softmax Loss는 score 값의 미미한 변화에도 민감하게 반응하여 성능을 더욱 높이려고 할 것이다.
즉, 정답 클래스 score 값이 충분히 높고 오답 클래스 score 값이 충분히 낮더라도, Softmax Loss는 score 값의 변화에 따라 Loss 값에 영향을 미치는 것이다.


지금까지의 내용을 정리해보면, 우리는 Linear Classifier를 통해 입력받은 data set의 score를 얻고, SVM Loss, Softmax Loss와 같은 손실 함수를 이용해서 모델의 예측값이 실제 정답 값에 비해 '얼마나 안좋은지'를 측정한다. 이후 손실 함수에 Regularization Term을 추가하여 모델의 '복잡함'과 '단순함'을 통제한다.
그렇다면 어떻게 실제 Loss를 줄이는 W를 찾을 수 있을까?
이 질문은 우리를 '최적화'라는 주제로 이끌어 준다.


## Optimization
최적화란 간단히 말해, Loss가 0인 지점을 찾아나서는 것이다.
우선 가장 먼저 생각해 볼 수 있는 단순한 방법은 임의 탐색(random search)이다.
임의 탐색이란 임의로 샘플링한 모든 W들을 모아놓고 Loss를 계산해서 어떤 W가 좋은지를 살펴보는 것이다.
물론 이는 정말 좋지 못한 알고리즘이고, 절대 이 방법을 사용해서는 안된다.

실제로 더 나은 전략은 지역적인 기하학적(local geometry) 특성을 이용하는 것이다.
이를 gradient descent(경사 하강법)이라 부른다.

![image](https://user-images.githubusercontent.com/37270069/113320822-eb0a0780-934d-11eb-8e13-cab73dac8e97.png)

경사 하강법에서는 우선 W를 임의의 값으로 초기화한다.
이후 Loss와 gradient를 계산한 뒤에 W를 gradient의 반대 방향으로 update 한다.
이는 gradient가 함수에서 증가하는 방향이므로, -gradient를 해야 내려가는 방향이 되는 것이다.
이에 대한 구체적인 방법은 chain rule과 함께 추후 강의에서 설명된다.
이 때 Step Size라는 hyper parameter는 -gradient 방향으로 얼마나 나아가야 하는지를 나타낸다.
이러한 Step Size는 Learning rate라고도 하며, 실제 학습을 할 때 고려해야 할 중요한 hyper parameter 중 하나이다.
또한 앞서 손실 함수에서의 최종 Loss는 전체 training data의 Loss 평균으로 계산되었다.

이 때 전체 training data의 크기가 굉장히 크다면 Loss를 계산하는 과정은 정말 오래 걸릴 것이며, W가 일일히 update 되려면 많은 시간이 걸리게 된다.
따라서 실제로는 stochastic gradient descent 라는 방법을 사용한다.

![image](https://user-images.githubusercontent.com/37270069/113320689-bd24c300-934d-11eb-9718-1cc3b3062550.png)

이는 전체 training data의 gradient와 loss를 계산하기 보다는, traininig data를 mini-batch라는 작은 training sample 집합으로 나누어서 학습하는 것이다.
따라서 이러한 작은 mini-batch를 이용해서 Loss의 전체 합에 대한 '추정치'와 실제 gradient의 '추정치'를 계산하는 것이다.
다시 말해, 임의의 mini-batch를 만들어내고, mini-batch에서의 loss와 gradient를 계산한 후, 이를 통해 W를 update 하는 것이다.


## Image Feature
지금까지 Linear Classifier에 대한 내용이었는데, 실제 Raw 이미지 픽셀을 입력으로 받는 방식이다.
하지만 이는 2장에서 말했던 것처럼 좋은 방법이 아니다.
이에 많은 연구가 진행되었고,
- 어떤 color 값이 많이 나오는지 count를 세서 특징을 추출하는 'Color Histogram'
- 방향값을 히스토그램으로 표현하고, 이미지의 8x8로 자른 후 해당 값에 어떤 각도가 많은지 히스토그램으로 나타내어 특징을 추출하는 'HoG(Histogram of Oriented Gradients)'
- 많은 이미지들을 가지고 이 이미지들을 임의로 잘라내어 k-means와 같은 알고리즘을 통해 군집화하여 각도, 색깔 등을 추출하고, 이러한 시각 단어(visual words)의 집합인 Codebook을 만들고 나면 새로운 이미지에서의 시각 단어들의 발생 빈도를 통해 이미지를 인코딩하는 'BOW(bag of words)'
등의 연구가 있었다.

이제는 특징을 뽑아내서 사용하는 것이 아닌 입력된 이미지에서 스스로 특징을 뽑아내도록 사용하고 있다.
CNN은 이미 만들어 놓은 특징들을 쓰기 보다는 데이터로부터 특징들을 직접 학습한다.
그렇기 때문에 raw 픽셀이 CNN에 그대로 들어가고 여러 layer를 거쳐서 데이터를 통한 특징 표현을 직접 만들어낸다.
따라서 Linear Classifier만 훈련하는 것이 아니라, 가중치 전체를 한꺼번에 학습하게 된다.
