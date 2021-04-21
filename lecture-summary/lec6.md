# Lecture 6 (Training Neural Networks I)
- 학습일: 2021/04/21
- 주제: One time setup, Training dynamics, Evaluation


## One time setup
### Activation Functions

![image](https://user-images.githubusercontent.com/37270069/115553918-e4d1d000-a2e8-11eb-8750-ded911ca7de4.png)  
Activation Function(활성화 함수)은 Wx에 대해 input이 들어오면, 이를 다음 노드로 보낼 때 어떻게 보낼지를 결정해주는 역할을 한다.  
이러한 Activation Function은 필수적인 요소이며, non-linear한 형태여야만 Network를 깊게 쌓을 수 있다.  

![image](https://user-images.githubusercontent.com/37270069/115554547-88bb7b80-a2e9-11eb-9584-20a351ad5946.png)  
활성화 함수에는 다양한 종류가 있으며, 각 함수에 따라 다음 layer로 전달되는 값의 형태 또한 다양하다.  

**1) Sigmoid**  
![image](https://user-images.githubusercontent.com/37270069/115557283-8575bf00-a2ec-11eb-9978-5045be1fe95b.png)  
Sigmoid 함수는 다음과 같은 3가지 단점으로 인해, 현재는 거의 사용되지 않는다.  
- Problem #1 : gradient vanishing  
- Problem #2 : not zero-centered  
- Problem #3 : compute expensive of exp()  

**2) Tanh**  
![image](https://user-images.githubusercontent.com/37270069/115557381-9faf9d00-a2ec-11eb-92b5-ca3de7255633.png)  
Tanh는 sigmoid의 단점을 일부 개선한 zero centered의 형태를 가지지만, 이 외의 2가지 단점인 gradient vanishing과 compute expensive of exp()로 인해 잘 사용되지 않는다.  

**3) ReLU**  
![image](https://user-images.githubusercontent.com/37270069/115557459-afc77c80-a2ec-11eb-914d-b8067e93893b.png)  
Relu는 가장 대중적으로 사용되는 Activation Function이나, not zero-centered와 0 이하의 값들은 모두 버려진다는 단점이 존재한다.  

**'Dead ReLU'**  
![image](https://user-images.githubusercontent.com/37270069/115557593-cf5ea500-a2ec-11eb-80bf-3096aa11ff46.png)  
위와 같이 입력으로 -10, 0이 들어온다면 gradient와 output 모두 0이 되어 업데이트가 되지 않는 'Dead ReLU' 상태에 빠지게 된다.  
이러한 Dead ReLU를 해결하기 위해 0.01의 bias를 주는 방법이 고안되었으나, 효능은 반반이라고 한다.  

**4) Leaky ReLU**  
![image](https://user-images.githubusercontent.com/37270069/115557935-28c6d400-a2ed-11eb-9c39-f1e2c82caa6d.png)  
Leaky ReLU는 0 이하의 값에 대해 작은 양수 값을 주어 기존 ReLU를 보완하였다.  
또한 이를 조금 변형한 것이 PReLU이며, 일종의 알파 값을 주고 학습을 통해 찾아가는 방식이다.  

**5) ELU**  
![image](https://user-images.githubusercontent.com/37270069/115558094-5875dc00-a2ed-11eb-919f-c1f2e3727f6c.png)  
ELU는 ReLU의 변형으로, 기존 ReLU의 장점을 모두 가지고 있으며 zero mean과 가까운 결과가 나오지만, compute expensive of exp()의 단점이 존재한다.  

**6) Maxout "Neuron"**  
![image](https://user-images.githubusercontent.com/37270069/115558344-9d017780-a2ed-11eb-8f88-addbb7083c8f.png)  
Maxout은 2개의 파라미터를 넘겨주고 max()를 이용해 더 좋은 값을 선택하는 방식으로, 연산량이 2배가 되기에 잘 사용되지 않는다.  

![image](https://user-images.githubusercontent.com/37270069/115558389-a7237600-a2ed-11eb-8021-8390ac201fc1.png)  
따라서 일반적으로 ReLU와 Leaky ReLU를 주로 사용하며, 필요에 따라 ELU, Maxout를 고려해보는 것도 좋다.  
또한 Tanh는 RNN과 LSTM에서는 자주 사용되지만, CNN에서는 사용되지 않는다.  
그리고 Sigmoid는 절대 사용되지 않는다.  

### Data Preprocessing  
![image](https://user-images.githubusercontent.com/37270069/115558868-1f8a3700-a2ee-11eb-85a1-c78cc569cc21.png)  
데이터 전처리를 위해서는 zero-centered, normalized가 주로 사용된다.  
zero centered는 양수, 음수를 모두 포함하는 행렬을 가질 수 있게 함으로써, 효율적으로 이동할 수 있다는 장점이 있다.  
normalized는 표준편차로 나눔으로써 데이터의 범위를 줄여 학습을 더 빨리 할 수 있고, local optimum에 빠지는 가능성을 줄여준다.  
이미지에서는 이미 [0,255]의 범위를 가지고 있기에, normalized는 잘 사용되지 않고 zero-centered를 많이 사용한다.  

![image](https://user-images.githubusercontent.com/37270069/115559415-a808d780-a2ee-11eb-8b68-ab8171ab6683.png)  
위와 같이 분산에 따라 차원을 감소시켜주는 PCA, whitened data 등도 존재하지만, 이미지에서는 잘 사용되지 않는다.  

### Weight Initialization  
Weight는 어떻게 초기화 되었는지에 따라 학습의 결과에 큰 영향을 줄 수 있다.  
![image](https://user-images.githubusercontent.com/37270069/115559930-18aff400-a2ef-11eb-88d2-d092183f3e62.png)  
만일 위와 같이 W가 0인 경우, output layer가 모두 0이 되고 gradient 또한 0이 되기 때문에 정상적으로 학습이 불가능하다.  

![image](https://user-images.githubusercontent.com/37270069/115560007-29606a00-a2ef-11eb-872d-449a5f2a28bd.png)  
그래서 작은 random 값을 weight에 설정해주어 사용하였고, 이는 작은 네트워크에서는 잘 동작하지만 깊은 네트워크에서는 잘 동작하지 않는다.  

![image](https://user-images.githubusercontent.com/37270069/115560802-e5ba3000-a2ef-11eb-9913-e98e465fcc9f.png)  
따라서 적절한 초기 weight 값을 주기 위해 Xavier init을 사용하였는데, 이는 노드의 개수를 normalization 하는 방법으로 input의 개수가 많아지면 크게 나눠주기 때문에 값이 작아지고, input의 개수가 적으면 weight 값이 커지는 방식으로 weight를 초기화하게 된다.  
Gradient Vanishing을 완화하기 위해서는 가중치를 초기화 할 때 출력 값들이 정규 분포 형태를 가져야 안정적인 학습이 가능하며, 이를 위해 입력의 분산과 출력의 분산을 동일하게 만드는 초기화 방식이다.  
이러한 Xavier는 Tanh와 같이 사용하기에는 좋지만, ReLU의 경우 출력 값이 0으로 수렴하고 평균과 표준 편차 또한 0으로 수렴하기에 Xavier를 사용하지 않는 것이 좋다.  
허나 이러한 점을 보완하기 위한 시도 결과, fan_in size를 2로 나누어 사용하면 ReLU에서 좋은 성능을 나타낸다.

### Batch Normalization  
![image](https://user-images.githubusercontent.com/37270069/115562936-f66ba580-a2f1-11eb-81ea-ddcc0c8a3fa9.png)  
Batch Normalization은 기본적으로 training 과정에서 Gradient Vanishing 문제가 일어나지 않도록 한다.  
이는 network의 각 층이나 activation 마다 input distribution이 달라지는 internal convariance shift를 방지하기 위해, 각 층의 input distribution을 평균 0, 표준편차 1인 분포로 만들어 internal convariance shift를 방지하는 것이다.  

![image](https://user-images.githubusercontent.com/37270069/115563166-36cb2380-a2f2-11eb-9b50-5f40639a84fe.png)  
보통 batch 별로 데이터를 train 시키는데, 이 때 NxD의 batch input이 들어오면 평균값을 빼주어 평균을 0으로 만들고, 분산을 나누어 분산값을 1로 만들어서 이를 normalize한다.  

![image](https://user-images.githubusercontent.com/37270069/115563384-7134c080-a2f2-11eb-95b4-c997db55e6f6.png)  
Batch Normalization은 일반적으로 activation layer 전에 사용되어 잘 분포되도록 한 후, activation을 진행할 수 있도록 하는데, 이는 Batch Normalization의 목적이 네트워크 연산 결과가 원하는 방향의 분포대로 나오는 것이기 때문에, activation function이 적용되어 분포가 달라지기 전에 적용하는 것이다.  
이러한 Batch Normalization을 사용할 때는 unit gaussian이 적합한지에 대해서 판단하여야 한다.  

![image](https://user-images.githubusercontent.com/37270069/115563845-e607fa80-a2f2-11eb-9b22-7698cd8a7c16.png)  
Batch Normalization이 적절한지에 대한 판단은 학습에 의해 조절할 수 있다.  
처음 normalize를 진행하고, 이 후 감마 값과 같은 하이퍼 파라미터 값들을 조정하여 Batch Normalization을 사용할지를 판단하는 것이다.  
위의 감마 값은 normalizing scale을 조절해주고, 베타 값은 shift를 조절해주는 파라미터 값이다.  
이 값들을 학습을 통해 조절함으로써 normalize 정도를 조절할 수 있다.  
또한 이러한 Batch Normalization을 사용하게 되면 overfitting을 완화해주기 위해 사용되는 Dropout을 대체할 수 있다고 한다.  

### Learning Process  
Step 1 : Preprocess the data  
Step 2 : Choose the architecture  
Step 3 : Double check that the loss is reanonable  
Step 4 : Let train using a small dataset  
Step 5 : Hyperparameter Optimization  

**Hyperparameter Optimization**  
![image](https://user-images.githubusercontent.com/37270069/115564973-ece33d00-a2f3-11eb-9b22-36543a212ea5.png)  
적절한 Hyperparameter를 찾기 위한 방법으로 Random Search와 Grid Search가 있다.  
Grid Search는 일정한 간격을 가지고 있어 best case를 찾지 못할 수 있다.  
반면, Random Search는 말 그대로 랜덤으로 떨어지기에 더 좋은 값의 영역에 접근할 확률이 높다.  
따라서 적절한 Hyperparameter를 찾기 위해 일반적으로 Random Search를 사용한다  
