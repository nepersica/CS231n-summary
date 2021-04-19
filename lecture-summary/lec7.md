# Lecture 4 (Introduction to Neural Network)
- 학습일: 2021/04/19
- 주제: Training Neural Network II

## Last Lecture
- Weight Initialization <br/>
: 가중치가 지나치게 작을 경우 모든 값이 0이 되며 학습은 일어나지 않음. <br/>
&nbsp; 반면, 가중치가 너무 큰 값으로 초기화되면 모든 값은 explode되는 일이 발생함.
- Data Preprocessing (zero-mean, unit variance 등)
- Batch Normalization<br/>
 : 미니 배치에서의 평균과 표준편차를 계산하여 Normalization을 수행함으로써 Trainig Dataset에 Overfitting을 방지함.
 - Hyperparmeter Search(Grid Search, Random Search)

Q. 보통 하이퍼 파라미터를 몇 개씩 선택하는가? <br>
→ 모델에 따라 다르다. 선택한 하이퍼 파라미터의 수가 많을수록 기하급수적으로 경우의 수가 늘어남.<br>
&nbsp; &nbsp;따라서 너무 많이 선택하는 것은 좋지 않음. 강연자의 경우 2~3가지, 많아도 4가지를 선택함.<br>
&nbsp; &nbsp; **중요도**: Learning rate > Regularization, decay, model size...

Q. 하이퍼 파라미터 값을 변경할 시에 다른 하이퍼 파라미터의 최적 값이 변하는 경우가 빈번한가?<br>
→ 가끔 발생한다. Learning rate가 좋은 범위 내에 속했으면 하지만 보통은 optimal보다 작은 값이고 학습 속도가 길어짐.<br>&nbsp; &nbsp;이런 경우 이번 강의에서 진행할 더 좋은 최적화 방법을 사용하면 모델이 learning rate에 덜 민감하게 함.


Q. Learning rate를 작게하고 epoch을 늘리면 어떤 일이 발생하는가?<br>
→ 동작은 하지만 학습하는데 엄청 오래 걸림. <br>
&nbsp; &nbsp; 실제로 learning rate가 0.001이냐 0.001이냐는 상당히 중요한 문제이다.

Q. Low learning rate를 할당하면 local optima에 빠질 수 있는가?<br>
→ 직관적으로 그럴 수 있지만 많이 발생하지는 않는다.
<br><br>

# This Lecture
- Facier Optimization
- Regularization
- Transfer Learning
<br><br>
## **Optimization**
### - **Stochastic Gradient Descent(SGD)** ~ 1차 미분을 활용한 방법<br>
&nbsp; &nbsp; 1. 미니 배치 안의 데이터에서 Loss를 계산<br>
&nbsp; &nbsp; 2. Gradient의 반대 방향을 이용해서 파라미터 벡터를 업데이트함.<br>
&nbsp; &nbsp; → 위 단계를 반복 수행하여 Optimization<br>

⭐ **문제점 1**<br>:
빨간 점에서 최적 점(이모티콘)으로 갈 때, 수평으로 이동하면 매우 느리고 수직으로 이동하면 비교적 빠르다.
손실 함수가 한 방향으로는 빠르게, 다른 방향으로는 천천히 변화되면 어떻게 될까? <br>
![image](https://user-images.githubusercontent.com/45097022/115173516-70d5d300-a102-11eb-86b4-474e333ee994.png)<br>
→ 위 그림과 같은 손실 함수를 가지고 있을 때 SGD를 수행하면 gradient의 방향은 고르지 못하여 손실함수가 지그재그 형태를 띈다.<br>
&nbsp; &nbsp; 손실 함수가 수직 방향으로만 Sensitive하기 때문에 지그재그로 이동하게 되어 high dimension일수록 더욱 심각해져서 큰 문제임.<br><br>

⭐ **문제점 2**:<br>
: x축은 어떤 하나의 가중치이고 y축은 손실함수를 나타낼때 SGD는 valley를 갖는 손실함수를 어떻게 움직이는가?
![image](https://user-images.githubusercontent.com/45097022/115178033-5fdd8f80-a10b-11eb-8818-ebdfa50e2b71.png)
<br>
→ Local minima : 멈춰버린다. valley에서 gradient가 0이 되어 학습이 멈추기 때문이다.<br>
→ Saddle points : 한쪽 방향으로는 증가하고 있고 다른 쪽 방향으로는 감소하고 있는 징겨 도한 gradient가 0이 된다.<br>
(Very large neural network는 local minima보다 saddle point에 취약함.)<br>
<br>
⭐ **문제점 3**:<br>
아래 예시의 N이 전체 training set일 경우에 N의 크기는 백만 단위의 크기가 될 수 있다.<br>
loss를 계산할 때마다 매번 모든 데이터 셋에 대해 계산하는 것은 어려움이 있음.
![image](https://user-images.githubusercontent.com/45097022/115185436-cd44ec80-a11a-11eb-879a-43cb63f10605.png)<br>
→ 따라서 실제로는 미니배치의 데이터들만 가지고 실제 loss를 추정한다.<br>
&nbsp; &nbsp; 각 지점의 gradient에 random uniform noise를 추가하고 SGD를 수행함.<br><br>

Q. SGD를 사용하지 않고 GD를 사용하면 문제점들이 해결되는가?
→ Full batch gradient descent에서도 동일한 문제가 발생한다.

✅ **해결 방안 1**<br>
SGD에 momentum term을 추가한다.<br>
**(momentum: velocity를 이용해서 step을 조절하는 방법)**
![image](https://user-images.githubusercontent.com/45097022/115185929-c5397c80-a11b-11eb-8d71-2b2e40718bfe.png)<br>
→ gradient를 계산할 대 velocity를 유지한다.<br>
&nbsp; &nbsp; velocity에 일정 비율 rho를 곱해주고 현재 gradient에 더하여 gradient vector의 방향이 아닌 velocity vector의 방향으로 나아간다.<br><br>

Q. 어떻게 SGD Momentum이 poorly conditioned coordinate 문제를 해결할 수 있는가?<br>
→ 하이퍼 파라미터인 rho의 영향을 받으면서 gradient는 계속해서 계산된다.<br>
&nbsp; &nbsp;  rho가 적절한 값으로 잘 동작한다면 velocity가 실제 gradient보다 더 커지는 지점까지 조금씩 증가하여 문제를 해결하는데 도움을 준다.<br><br>

## **SGD Momentum**
![image](https://user-images.githubusercontent.com/45097022/115186641-0b431000-a11d-11eb-8a89-1dc2fd13d07a.png)<br>
- 기본 SGD momentum
  - Red vector: 현재 지점에서의 gradient의 방향
  - Green vector: Velocity vector
    → 실제 업데이트는 Red와 Green vector의 가중 평균으로 구하여 gradient의 noise를 극복할 수 있게 한다.<br>
     (기본 SGD momentum은 "현재 지점"에서의 gradient를 계산한 뒤 velocity와 섞어 준다.)<br>

- Momentum의 변형= **Nesterov accelerated gradient**<br>
    - 빨간 점에서 시작해서 우선은 Velocity 방향으로 움직이고 그 지점에서의 gradient를 계산한다. 그리고 다시 원점으로 돌아가 둘을 합친다.<br>
     (두 정보를 약간 더 섞어준다고 생각하면 편하다.)<br>
     → Convex optimization 문제에서는 뛰어난 성능을 보이지만 Neural Network와 같은 non-convex problem에서는 성능이 보장되지 않는다.
<br>
<br>

Q. velocity의 초기값을 구하는 좋은 방법이 있는가?
→ velocity은 하이퍼 파라미터가 아니며 초기값은 항상 0이다.
→ velocity = gradient의 weighted sum


- Nesterov Momentum 수식
![image](https://user-images.githubusercontent.com/45097022/115187255-f5821a80-a11d-11eb-8ba9-b51594a75bc3.png)<br>
첫 번째 수식은 기존의 momentum과 동일하게 velocity와 계산한 gradient를 일정 비율로 섞어주는 역할을 한다.<br>
두 번째 맨 밑의 수식은 현재 점과 velocity를 더하고 "현재 velocity - 이전 velocity"를 계산하여 일정 비율(rho)를 곱하고 더해준다.<br>
→ Nesterov momentum은 현재/이전 velocity 간의 에러 보정이 추가되었다.

![image](https://user-images.githubusercontent.com/45097022/115187839-f49db880-a11e-11eb-8a17-69972e665f40.png)<br>
- SGD: loss에 수렴하는데 오랜 시간이 소요된다.
- SGD + Momentum: 이전의 velocity의 영향으로 인하여 minima를 지나치지만 스스로 경로를 수정하고 결국 inima에 수렴한다.<br>
- Nesterov: 일반 momentum에 비해서 overshooting이 덜하다.
  

Q. 위 예시로는 momentum의 성능이 좋아보이는데 만약 minima가 엄청 좁고 깊은 곳이면 어떻게 되는가?<br>
&nbsp; &nbsp;&nbsp; momentum의 velocity가 오히려 minima를 건너 뛰는 현상도 발생할 수 있지 않는가?<br>
→ 해당 내용은 최근의 연구들이 주목하는 주제이다.<br>
&nbsp; &nbsp; 하지만 좁고 깊은 minima는 훨씬 더 심한 overfits를 불러오기 때문에 좋은 minima가 아니다.<br>
&nbsp; &nbsp; Training data가 더 많이 모이면 그런 민감한 minima는 점점 사라져 평평한 minima를 얻게 된다.<br>
<br>


### - **AdaGrad** <br>
![image](https://user-images.githubusercontent.com/45097022/115188522-ff0c8200-a11f-11eb-9e1f-996f53c2611c.png)<br>
: 훈련 중 계산되는 gradients를 활용하는 방법으로, velocity term 대신 grad squared term을 이용한다.<br>
&nbsp; &nbsp; 학습 도중에 계산되는 gradient에 제곱을 해서 계속 더해준다.<br>
&nbsp; &nbsp; 그리고 update를 할 때 update term을 앞서 계산한 gradient 제곱 항으로 나눠준다.<br>


⭐ **문제점 1**<br>
: 학습 횟수 t가 계속 늘어나면(학습이 계속 진행되면) 어떻게 되는가?<br>
→ step을 진행할수록 값이 점점 작아진다.<br>
&nbsp; &nbsp; update 동안 gradient의 제곱이 계속해서 더해져 서서히 증가되어 step size를 점점 더 작은 값이 되게 한다.<br><br>

### - **RMSProp** <br>
![image](https://user-images.githubusercontent.com/45097022/115188945-a8ec0e80-a120-11eb-8188-5b6f22fd8133.png)<br>
AdaGrad의 gradient 제곱항을 그대로 사용한다.<br>
하지만 RMSProp는 이 값들을 그저 누적하는 것이 아니라 기존의 누적 값에 decay_rate를 곱해준다.<br>
이는 점점 속도가 줄어드는 문제를 해결할 수 있다.<br><br>

**Adam** : momentum 계열과 Ada 계열을 조합한 결과<br>
![image](https://user-images.githubusercontent.com/45097022/115189227-1f890c00-a121-11eb-823b-ff4ac2ed0340.png)<br>
- first momentum: gradient의 가중 합<br>
- second momentum: AdaGrad나 RMSProp처럼 gradients의 제곱을 이용하는 방법.<br>
&nbsp; &nbsp; 거의 모든 아키텍쳐에서 잘 동작하는 기본 설정으로 좋다.<br>

Q. 수식 10^-7이 무엇인가?<br>
→ 분모에 작은 양수 값을 더해줘서 0이 되는 것을 사전에 방지한다.<br>
&nbsp; &nbsp; 하이퍼 파라미터이긴 하나 영향력은 없다.<br>

![image](https://user-images.githubusercontent.com/45097022/115189598-ab029d00-a121-11eb-91c5-e8e594faa154.png)<br><br>


![image](https://user-images.githubusercontent.com/45097022/115189710-e3a27680-a121-11eb-8132-4ce5d0f9c71e.png)<br>
ResNet 논문은 **step decay learning rate**를 사용하여 gradient가 평평해지다가 갑자기 내려가는 구간은 learning rate를 낮춘다.<br>
<br>
learning rate decay를 설정하는 순서는 우선 decay 없이 학습을 시킨 후, loss curve를 분석하여 decay가 필요한 곳을 고려한다.<br><br>

〓> Optimization 알고리즘들은 training error를 줄이고 손실함수를 최소화하기 위한 역할을 수행한다.

그렇다면, 처음 보는 데이터에 대한 성능을 끌어올리기 위해서는 어떻게 해야 할까? → **Model Ensemble(모델 앙상블)**<br>

### **Model Ensemble(모델 앙상블)**<br>
1. 모델을 하나만 학습시키지 않고 10개의 모델을 **독립적으로 학습**시킨다. 그리고 10개 모델 결과의 평균을 이용하는 방법
2. 학습 도중 중간 모델들을 저장(snapshots)하고 앙상블로 사용한다. 그리고 Test time에는 여러 snapshots에서 나온 예측값들을 평균내서 사용한다.<br><br>

Q. 모델 간의 loss 차이가 크면 한 쪽이 overfitting이 될 수 있으니 별로 안좋고 차이가 작아도 안좋지 않은가?<br>
→ 그래서 좋은 앙상블 결과를 얻기 위해 모델 간의 최적의 갭을 찾는 것이 중요하다.<br>
사실 중요한 것은 gap이 아닌 **validation set의 성능을 최대화**시키는 것이다.<br>

Q. 앙상블 모델마다 하이퍼 파라미터를 동일하게 줘야하는거?<br>
→ 그렇지 않을 수 있다. <br>
&nbsp; &nbsp; 다양한 모델 사이즈, learning rate 그리고 regularization 기법 등을 앙상블할 수 있다.
<br>&nbsp; &nbsp; 또는 학습하는 동안에 파라미터의 exponentially decaying average를 계속 계산한다.
<br>
<br>

### **Regularization**<br>
![image](https://user-images.githubusercontent.com/45097022/115190737-768fe080-a123-11eb-90e2-f07065e64ef5.png)<br>
L2 Regularization은 Neural Network에서 사용하지 않는다.<br>

## **Dropout**<br>
forward pass 과정에서 임의로(랜덤으로) 일부 뉴런을 0으로 만든다.<br>
![image](https://user-images.githubusercontent.com/45097022/115190963-bd7dd600-a123-11eb-867d-aa274c151803.png)<br>
Dropout은 한 레이어씩 진행한다. ~ 먼저 한 레이어의 출력을 전부 구하고 임의로 일부를 0으로 만들어 다음 레이어로 넘어간다.<br><Br>

Q. 무엇을 0으로 설정하는 것인가?<Br>
→ Activation을 0으로 설정한다.<br>
&nbsp; &nbsp; 각 레이어에서 next_active = prev_active * weight이다.<br>
&nbsp; &nbsp; activation의 일부를 0으로 만들면 다음 레이어의 일부는 0과 곱해진다.<br>
<br>
Q. 어떤 종류의 레이어에서 사용하는가?<br>
→ FC layer에서 흔히 사용하나 conv layer에서도 종종 볼 수 있다.<br>
&nbsp; &nbsp; conv net에서는 전체 feature map에서 dropout을 수행한다.<Br>
&nbsp; &nbsp; conv layer는 여러 channel이 있기 때문에 일부 channel 자체를 dropout시킬 수 있다.<br>
<Br>
Q. Dropout을 사용하게 되면 train time에서 gradient에는 어떤 일이 일어나는가?<br>
→ Dropout이 0으로 만들지 않은 노드에서만 backprop이 발생하게 된다.

〓> Dropout을 사용하게 되면 전체 학습시간은 늘어나지만 모델이 수렴한 후에는 더 좋은 일반화 능력을 얻을 수 있다.<br><Br>
![image](https://user-images.githubusercontent.com/45097022/115191674-b73c2980-a124-11eb-8698-250162c53b1f.png)<Br>
train time에 레이블은 그대로 놔둔채로 이미지를 무작위로 변환시킨다.<br>


## **DropConnect**<br>
![image](https://user-images.githubusercontent.com/45097022/115191804-e3f04100-a124-11eb-984a-1ffc412ce562.png)<br>
: activation이 아닌 weight matrix를 임의로 0으로 만들어준다.<br><br>

Q. 보통 하나 이상의 regularization을 사용하는가?
→ 일반적으로 대부분의 네트워크에서 잘 동작하는 batch normalization을 사용한다.