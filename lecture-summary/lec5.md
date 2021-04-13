# Lecture 5 (Convolutional Neural Networks)
- 학습일: 2021/04/13
- 주제: Backpropagation, Neural Networks

## Last time
1. Nerual Network와 선형 함수, 비선형 레이어(Nonlinearities = Softmax, etc)
2. NN은 Mode 문제를 해결할 수 있다!  
→ 자동차를 분류하는 문제도 풀 수 있다

## Convolutional Nerual Network
- 공간적 구조를 유지하는 네트워크임

### Historical Facts - Phase I "NN to CNN"
- 1957년도 Mark I의 Perceptron으로부터 시작함
- 1960년도 Adaline and Madaline으로 Multilayer Perceptron Network로 발전
- 1986년도 Backprob 제안
- 2006년도 Deep Nerual Network를 효과적인 학습 가능성 다시 밝혀짐
- 2012년도 Speech Recognition과 ImageNet Classification에서 높은 성능을 보임 (AlexNet)

### Historical Facts - Phase II "Convolutional Nerual Nets"
- 1950년도 고양이에 대한 시각 피질 뉴런의 연구  
→ 뇌가 Oriented Edge와 Shape에 반응하였음  
→ 뉴런이 계층 구조를 띄고 있는 것을 확인함  
→ 계층이 점점 깊어짐에 따라서 Orientation에서 Movement로 변하는 등 복잡한 Feature에 반응하였음
- 1980년도 Neocognition이 최초의 아이디어 구현체
- 1998년도 LeCun의 문서 인식 구현도 아주 잘 작동함
- 2012년도 AlexNet으로 ImageNet과 같은 대규모 데이터를 활용할 수 있게 되었음 + GPU!

### ConvNets - Usages
- 각종 Task가 가능함 - Image Classification, Detection, Segmentation, Pose Estimation
- Reinforcement Learning을 이용한 게임 플레이 학습도 가능
- Image Captioning도 가능
- GAN을 이용한 예술 작품도 가능!
- Style Transfer를 이용하여 특정 화풍으로 다시 그려줄 수도 있음

### ConvNets - How does it works?
- FC 레이어에서 하는 일은 "벡터를 가지고 연산하는 것"이다!
- Weight와 입력 Feature Vector를 Matrix Multiplication하여 그 결과를 Layer Output으로 가진다.

  ![image](https://user-images.githubusercontent.com/5201073/114549877-854a4380-9c9c-11eb-90f4-fc2be8921ecb.png)
- ConvLayer는 FC와는 다르게 "기존의 구조를 보존한다".
- Window(=Kernel)를 Sliding하면서 공간 내적을 수행한다.
- 32x32x3의 입력 이미지를 Sliding하여 내적한 결과 28x28의 Output Feature가 나온다.
- 필터의 갯수가 n=6개라면, Output Feature는 n=6의 채널 수를 가지게 된다.
- 이제 이 개념을 가지고 네트워크를 만들게 된다.

  ![image](https://user-images.githubusercontent.com/5201073/114550118-d0645680-9c9c-11eb-9969-59e35486d5ce.png)
- 이제 Conv-ReLU-(Pooling sometimes)가 들어가게 된다.
- (다음 수업에서는 Visualization에 대해서 해보도록 한다!)
- 이 내용은 역시 Hubel/Wiesel의 이론과도 동일하게 앞에서는 단순한 내용을, 뒤에서는 복잡한 내용을 처리하게 됨

  ![image](https://user-images.githubusercontent.com/5201073/114550447-2d600c80-9c9d-11eb-998d-df2fed59eddf.png)
- Activation Map은 각 Weight를 거쳐 나온 Feature Map을 의미한다! (Weight 자체의 값이 아니다!)
- 위의 각 Activation을 보면 필터가 Edge를 찾고 있는 것을 확인할 수 있다.

  ![image](https://user-images.githubusercontent.com/5201073/114550724-7adc7980-9c9d-11eb-9ca6-b5ed84840a71.png)
- (32x32x3)의 Feature를 (5x5x3)의 필터를 가지고 연산하면 (28x28x3)의 Output이 나오는 이유?  
→ 가로 방향으로 봤을 때, 필터 기준으로 왼쪽 2개, 오른쪽 2개가 손실되므로 총 4개의 픽셀이 빠진다. (세로도 마찬가지)
- Stride를 이용하여 Sliding Window의 건너뛰는 정도를 설정할 수 있다. (커질수록 Output Feature가 작아진다)  
→ 출력을 `(N - F) / stride + 1`로 계산할 수 있다! 여기서 소수점 값이 나온다는 것은 잘 작동하지 않는다는 의미이다  
(잘 잘리지 않아 손실이 있을 것이다.)

- **보통 사용하는 방법**: 3x3s1, 5x5s2, 7x7s3
- Feature가 작아지면 작아질수록 좋지 않다 (많은 정보를 잃어버린다)
- **보통 사용하는 필터의 갯수**: 2의 제곱수 (2, 4, 8, 16, 32, 64, ...)

#### 1x1 Conv!
- 슬라이딩하면서 똑같이 값을 구함 but Depth에 대한 부분만
- Depth에 대한 내적이므로 다른 채널들(Depth들)에 대한 상관관계를 (모델이) 신경쓰게 되는 것이다.

#### Reception Field
- 한 뉴런(=Kernel)이 한 번에 수용할 수 있는 영역을 의미함


#### 정리
- 결국 FC는 전체 정보를 이용하지만 ConvNet은 지역적인 정보만 이용한다.

### Pooling Layer
- Feature를 더 작게 해준다! ("관리하기 쉬워진다"라고 표현함)
- 목적은 공간적인 불변성과 계산하는 Parameter 수의 감소이다.
- Depth에는 아무것도 하지 않고, 공간적인 크기만 줄인다.

  ![image](https://user-images.githubusercontent.com/5201073/114552018-15898800-9c9f-11eb-9dd9-d13fced91869.png)
- Max Pooling을 하면 kernel 안의 공간 중 가장 큰 값만 가지게 된다.
- **Max pooling을 쓰는 이유?**: Activation이라는 것은 Neruon이 얼마나 활성화되어있는지를 의미하는 것이다.  
→ 값이 얼마나 큰 지가 곧 Activation이므로, 값의 크기와 그 큰 값이 위치한 픽셀공간의 위치가 중요한 직관이 된다.
- ** 보통 사용하는 세팅**: 2x2, 3x3, stride=2 (3x3일 때에도!) but 2x2가 역시 잘 쓰임

### FC Layer
- 전체 ConvNet 뭉치들을 거친 뒤에는 FC Layer에 입력하기 위해 전체를 1차원 Vector의 형태로 편다(Flatten).