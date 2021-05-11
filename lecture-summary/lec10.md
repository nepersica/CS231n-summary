# Lecture 10
- 학습일: 2021/05/10
- 주제: Recurrent Neural Networks

## Last Lecture
- CNN Architecture<br/>
    - AlexNet <br/> 
    - VGG / GoogleNet ~ Batch Normalization이 존재하지 않았던 당시 깊은 모델을 학습시키는데 어려움이 존재<br/>
    - ResNet ~ short cut connection과 residual block이 도입된 모델<br/>
        - identity mapping in Deep Residual Network(resnet이 좋은 성능이 나오는 것을 검증한 논문)
        - L2 Regularization을 사용하여 과적합이 되지 않도록 weight를 조정
    - DenseNet, FractalNet ~ 모델 내부에 additional shortcut을 추가한 모델<br/>

![image](https://user-images.githubusercontent.com/45097022/117643680-9b77f080-b1c3-11eb-8b23-23aa669f5d3e.png)
- AlexNet과 ResNet은 FC layer로 인하여 파라미터 개수가 큼<br/>
- GoogLeNet과 ResNet 등은 FC layer 대신 GAP(Global Average Pooling)으로 대체하여 파라미터 개수 감소<br/><br/><br/>

# This Lecture
- Recurrent Neural Network
<br><br>
## **"Vanilla" Nerual Network**
### : 현재까지 배운 아키텍처는 **"one to one" model**로, 단일 입력(영상)을 받아 hidden layer를 거쳐 단일 출력<br/>
![image](https://user-images.githubusercontent.com/45097022/117644090-17723880-b1c4-11eb-86cf-9b491193f28c.png)<br>
모델은 다양한 입력을 처리할 수 있도록 유연해야 함. <br/>
- **"one to many" model** : 단일 입력(영상)을 받아 가변 출력(caption)<br/>
- **"many to one" model** : 가변 입력(문장)을 받아 단일 출력(감정)
- **"many to many" model** 
  - 가변 입력(영어 문장)을 받아 가변 출력(프랑스어 문장)
  - 가변 입력(동영상)을 받아 가변 출력(매 프레임의 classificatio 결과)<br/>
=> RNN은 위와 같은 다양한 상황들을 모델이 잘 처리할 수 있도록 함<br/>
    +) 이미지 생성 측면에서도 매우 뛰어남.

## **RNN(Recurrent Neural Network)**<br/>
→ 입력을 받아 hidden state를 거쳐 모델에 feedback되고(update) 출력을 내보냄. 그리고  새로운 입력 x를 받아 동일 과정을 수행함.<br/>
    +) hidden state: RNN이 새로운 입력을 불러들일 때마다 매번 업데이트됨.<br/>
<br/>
### RNN을 수식으로 나타낸 그림<br/>  
![image](https://user-images.githubusercontent.com/45097022/117646948-53f36380-b1c7-11eb-850d-71ab971bfb79.png)<br/><br/>
![image](https://user-images.githubusercontent.com/45097022/117647312-c7957080-b1c7-11eb-89e6-c7cd163f7b68.png)
<br/>
- 초록색 RNN block: 함수 f로 **재귀적인 관계**를 연산할 수 있도록 설계<br/>
  - 함수 f는 "이전 상태의 hidden state인 h_t-1"과 "현재 상태의 입력인 x_t"를 입력으로 받아 h_t(x_t+1) 출력<br/>
<br/>

## 1. Many to Many model 수행
![image](https://user-images.githubusercontent.com/45097022/117783328-64661580-b27d-11eb-8a2c-b11382f8fd07.png)
<br/>
- 과정 ~ hidden satte를 갖고 "재귀적"으로 feedback → Multiple time steps를 unrolling하여 입/출력, 가중치 행렬들 간의 관게를 더 명확하게 이해하기<br/>
    : 첫 step에서 h_0(hidden state))를 0으로 초기화, x_1(입력)과 h_0를 함수 f_w의 입력으로 넣어 h_1(출력) 획득 ~> 반복 <br/>
    => 이때, 동일한 가중치 행렬 W가 매번 사용됨<br/>

    backward pass시 dLoss/dW를 계산하려면 행렬 W의 gradient를 전부 더해줘야 함. <br/>
    ~> 이를 위한 행렬 gradient는 각 스텝에서의 W에 대한 gradient를 전부 계산한 뒤, 이 값들을 모두 더함.
- RNN의 출력(h_t)는 또 다른 네트워크의 입력으로 들어가 y_t(class score)를 생성
- Loss = softmax loss, 최종 Loss = 각 loss들의 합<br/>
    +) softmax 함수: 출력을 0~1 사이의 값으로 모두 정규화하여 총합이 1이 되게하는 함수<br/>
- Back propagation으로 모델을 학습시키기 위해 dLoss/dW를 구해야 함.<br/>
  
<br/>    

## 2. Many to One model 수행 <br/>
ex) 감정 분석<br/>

![image](https://user-images.githubusercontent.com/45097022/117783922-f706b480-b27d-11eb-8336-d3635f8a32c3.png)
<br/>
=> 네트워크의 최종 hidden state에서만 결과 값이 나옴. (ㅊ전체 시퀀스의 내용에 대한 일종의 요약을 볼 수 있기 때문)<br/>    

## 3. One to Many model 수행<br/>
![image](https://user-images.githubusercontent.com/45097022/117784148-3503d880-b27e-11eb-9c4d-fa730e9533f4.png)<br/>
- "고정 입력"을 받아 "가변 출력"하는 네트워크 => 모델의 initial hidden state를 초기화시키는 용도로 사용<br/>
- 모든 스텝에서 출력 값을 가짐<br/>

<br/>

## 4. Sequence to Sequence model 수행<br/>
![image](https://user-images.githubusercontent.com/45097022/117784316-67add100-b27e-11eb-9ce2-455c47ef7397.png)<br/>
 (Many-to-one model) + (one-to-many) => Encoder + Decoder
- Encoder → 가변 입력(ex. English sentence)를 받아 final hidden state를 통해 전체 sentence를 하나의 벡터로 요약<br/>
- Decoder → 입력(벡터)를 받아 가변 출력(다른 언어 sentence, 매 스텝에 적절한 단어)를 출력
- Output sentence의 각 loss들을 합하여 back propagation 수행<br/>

→ Language modeling에서 자주 사용함.<br/>
- ex) Character level language model<br/>
        : 네트워크는 문자열 sequence를 읽어들여 현재 문맥에서 다음 문자를 예측함.<br/><br/>
        ![image](https://user-images.githubusercontent.com/45097022/117785643-bf990780-b27f-11eb-926d-2b935c6e16f1.png)<br/><br/>

- Test Time의 모델 수행 과정<br/>
    → Train Time의 모델이 예측했을 문장을 모델 스스로 생성함.<br/>
    예를 들어, 'h'가 주어지면 모든 문자에 대한 스코어를 획득하고 softmax를 통해 확률 분포로 표현된 스코어를 **sampling**(다음 글자 선택)에 이용함.<br/><br/>

    
    Q. 가장 높은 스코어를 갖는 클래스를 사용하면 되는데 왜 softmax를 한 결과로부터 샘플링을 하는가?<br/>
    → 실제로 모두 사용 가능하다. 하지만, 확률분포에서 샘플링하는 방법을 사용하면 모델에서의 다양성을 얻을 수 있다.<br/>
    샘플링하는 방법은 다양한 출력을 얻을 수 있다는 관점에서 좋은 방법이다.<br/>

    Q. Test time에 softmax vector를 one hot vector 대신 넣어줄 수 있는가?
    → one hot을 사용하지 않으면 두 가지 문제점이 발생한다.
    1. 입력이 train time에서의 입력과 달라진다.
    2. 실제 vocabulary의 크기는 매우 크다<br/>
    실제로 one hot vector를 sparse vector operation으로 처리한다.<br/><br/>

![image](https://user-images.githubusercontent.com/45097022/117787889-d8a2b800-b281-11eb-8a30-d33b2aa1dab5.png)<br/>

- 출력값들의 loss를 계산해 final loss를 획득한다. => backpropagation though time
- 실제로 truncated backpropagation을 통해 back prob을 근사시키는 기법을 사용한다. -> sequence가 매우 길어도 한 스텝을 일정 단위로 자른다.
  - Train time에서 step을 100으로 자르고 해당 스텝의 sub sequence의 loss를 계산하여 gradient step을 진행하는 과정을 반복한다.
  - 다음 batch의 forward pass를 계산할 때는 이전 hidden state를 이용한다.<br/>
   
- => 비전에서 Image Classification을 할때 Large scale의 데이터를 mini batch만을 이용하여 gradient를 계산하는 것과 비슷함.<br/>
  
### Truncated Backpropagation <br/>
: very large sequence data의 gradient를 근사시키는 방법
<br/><br/>

## **Image Captioning model** <br/>
: 입력은 이미지, 출력은 자연어로 된 Caption(가변 길이)인 모델<br/>
![image](https://user-images.githubusercontent.com/45097022/117789735-98443980-b283-11eb-9946-3860162ddf7c.png)<br/>
ex) **Attention**: Caption을 생성할 때 이미지의 다양한 부분을 집중(attetion)하여 caption을 출력함.<br>
<br/>

## **Multi-layer RNN** 
![image](https://user-images.githubusercontent.com/45097022/117790168-11439100-b284-11eb-837d-7e6e69e8a938.png)<br/>
→ 모델이 깊어질수록 다양한 문제들에서 성능이 좋아진다. 하지만 보통 깊은 RNN 모델을 사용하지 않는다. 2~4 layer가 가장 적절하다.<br/><br/>

RNN을 학습시킬때 문제가 발생하기 때문에 잘 사용하지 않는다.<br/>

![image](https://user-images.githubusercontent.com/45097022/117791022-de4dcd00-b284-11eb-8cf7-52b487c1a9c5.png)<br/>
→ RNN 기본 수식을 사용하면 많은 가중치 행렬들이 개입하게 되어 graident vanishing이 발생하게 되어 매우 비효율적이다.

## **LSTM(Long Short Term Memory)**
![image](https://user-images.githubusercontent.com/45097022/117791722-924f5800-b285-11eb-897b-396352710fb3.png)<br/>
LSTM은 두 개의 hidden satte가 존재하고 c_t라는 내부에만 존재하는 변수가 존재한다.<br/><Br/>

![image](https://user-images.githubusercontent.com/45097022/117791570-6d5ae500-b285-11eb-925f-9f1b3dffbcd8.png)
- 입력(i, x_t에 대한 가중치), 이전 cell에 대한 정보를 지우는 f, 출력(o), gate(g)로 구성되어 있다.

### LSTM Flow Chart 
![image](https://user-images.githubusercontent.com/45097022/117792019-e0fcf200-b285-11eb-9181-4e57268417ee.png)<br/>
: 하나의 cell에서 Weight 행렬을 거치지 않고 cell 단위에서 gradient 계산을 하여 여러 cell을 거쳐도 계산량이 크게 증가하지 않는다.
=> f와 곱해지는 연산이 행렬 단위의 연산이 아니라 element-wise이기 때문이다.
  
