# Lecture 4 (Introduction to Neural Network)
- 학습일: 2021/04/02
- 주제: Backpropagation and Nerual Networks

## Computational graph
### input이 x,W이며 Regularization 항을 가진 선형 classifier 예제 그래프
![image](https://user-images.githubusercontent.com/45097022/113536015-fd1acd00-960f-11eb-9777-330cb75561e5.png)

- 그래프의 각 노드= 연산 단계

→ Computational graph를 사용해서 함수를 표현하게 됨으로써 **Backpropagation**을 사용할 수 있게 되었다.<br/><br/>

## **Backpropagation**
![image](https://user-images.githubusercontent.com/45097022/113545873-f008d880-9625-11eb-8d90-6830886d83a3.png)
gradient를 얻기 위해 computational graph 내부의 모든 변수에 대해 뒤에서부터 chain rule을 재귀적으로 사용한다.<br/><br/>

![image](https://user-images.githubusercontent.com/45097022/113547554-1d0aba80-9629-11eb-9c58-bf5d0aceac90.png)
 - 각 local 노드의 input = x,y / output = z일 때, 미적분을 통하여 z에서 x나 y에 대한 gradient를 구할 수 있다.
 - 예를 들어, x의 gradient는 z에 대한 gradient와 x에 대한 z의 local gradient로 합성된다.
→ Backpropagation과 Chain rule을 통해 필요한 gradient를 계산할 수 있다.<br/><br/>

+) Sigmoid Function의 Backpropagation
![image](https://user-images.githubusercontent.com/45097022/113548061-dcf80780-9629-11eb-9f93-a2164bed7530.png)
<br/><br/>

## Patterns in backward flow
![image](https://user-images.githubusercontent.com/45097022/113549005-8db2d680-962b-11eb-9c14-0049f825e125.png)
<br/>
위 이미지에서,
- max function에 대한 gradient는 어떻게 될것인가?<br/>
→ z는 gradient 2를 갖고 w는 gradient 0를 갖는다. max function은 단순히 gradient가 통과하는 효과를 갖는다.(=gradient router)<br/>
- mul function에 대한 gradient는 어떻게 될것인가?<br/>
local gradient는 기본적으로 다른 변수의 값이다. upstream gradient를 받아 다른 브랜치의 값으로 scaling한다.(=gradient switcher) 

=> gradient가 주어지면 가중치를 업데이트 하기 위해 다음 스텝을 진행하여 Optimization할 때, Backpropagation을 사용하여 신경망과 같이 임의의 복잡한 함수에 대해 gradient를 계산할 수 있다.
<br/><br/>

 x,y,z가 **벡터**이면 gradient는 Jacobian 행렬이 된다. (Jacobian 행렬: 원소가 모두 1차 미분 계수로 구성된 행렬)<br/>
ex) x의 각 원소에 대해 z에 대한 미분을 포함하는 행렬
![image](https://user-images.githubusercontent.com/45097022/113550147-8bea1280-962d-11eb-901c-486e8f4ecfc8.png)<br/><br/>

## Vectorized operations<br/>
input이 4096-d vector이고 요소별로(elementwise) max function을 수행하였을 때 출력 또한 4096-d vector이다.

![image](https://user-images.githubusercontent.com/45097022/113550348-e2575100-962d-11eb-8331-7ba25955bdcd.png)
<br/>
이러한 경우, Jacobian 행렬의 사이즈는 4096^2이다. ~> Jacobian 행렬의 각 행은 입력에 대한 출력의 편미분이다.<br/>
+) minibatch가 100이면 Jacobian 행렬의 사이즈는 4096000^2이다. 이는 비효율적이지만 실제로 이렇게 큰 사이즈의 Jacobian 행렬을 계산할 필요가 없다.<br/><br/>

## Vectorized Example
n차원을 갖는 x와 n*m 차원을 갖는 W에 대한 함수 f가 x에 의해 곱해진 W의 L2와 같다.<br/>
 
![image](https://user-images.githubusercontent.com/45097022/113552303-f18bce00-9630-11eb-98ae-71065eaca7df.png)
위 이미지를 참고하여 f를 2차원의 벡터인 q에 대한 표현으로 나타내면 q에 대한 L2로 나타낼 수 있다 => q^2_i <br/>
그리고 f에 대해 q^2_i를 미분하면 2q_i를 획득한다.
x_j와 동일한 W_i,j에 대한 q_k의 식으로 W에 대한 q의 local gradient를 일반화할 수 있다.

+) 각 W_i,j에 대해서 f에 대한 gradient는 Chain Rule을 사용하여 구할 수 있다.<br/>
q_k에 대한 f의 미분을 합성할 때, dq_k/W_i,j에 대하여 Chain Rule을 수행하여 W와 q의 각 요소들의 영향을 찾을 수 있다.<br/>
이는 결국 2* q_i* x_j와 동일한 식을 갖는다.<br/><br/>
 
![image](https://user-images.githubusercontent.com/45097022/113562053-716d6480-9640-11eb-9397-c792171104cd.png)
위 이미지는 x_i에 대한 q와 f의 local gradient를 일반화하는 식에 대해 풀이되어있다.

☞ 각 노드를 local하게 보았을 때, upstream gradient와 함께 chain rule을 이용하면 Local gradient를 구할 수 있다.
<br/><br/>

## forward pass, backward pass의 API
![image](https://user-images.githubusercontent.com/45097022/113562315-d9bc4600-9640-11eb-9132-311459fcc793.png)
<br/>

- forward pass: 노드의 출력을 계산하는 함수를 구현
- backward pass: chain rule을 이용하여 gradient를 계산하는 함수를 구현

## Deep Learning Framework<br/>
Caffe,, => 라이브러리에서 layer에 대하여 구현된 코드들을 볼 수 있음.<br/><br/>

# Neural Network<br/>

## **신경망**은 **복잡한 함수를 만들기 위해서 간단한 함수들을 계층적으로 쌓아올린 함수들의 집합**이다.

![image](https://user-images.githubusercontent.com/45097022/113563623-feb1b880-9642-11eb-934c-4756594fa42a.png)

- Linear Score Function: f=Wx<br/>
- 2-layer Neural Network: f=W_2*max(0, W_1x) → 가중치 W1과 입력 x의 행렬 곱 h을 얻고 max(0,W)의 비선형 함수를 이용하여 선형 레이어 출력의 max를 얻는다.<br/><br/>

    이때 h는 현재 템플릿들에 대한 모든 스코어를 갖고 있고, h를 결합하는 또 다른 레이어를 가질 수 있다.

    Q.W2가 가중치를 갖고 있는 것인가, h가 가중치를 갖고 있는 것인가?
    → h는 W1에서 가지고 있는 템플릿에 대한 스코어 함수이다. W2가 템플릿에 가중치를 부여하고 모든 중간 점수를 더해 클래스에 대한 최종 점수를 얻는다.
    <br/><br/>
- 3-layer Neural Network: f=W_3 * max(W_2*max(0, W_1x))<br/><br/>
  
## Activate Functions(활성함수)<br/>
![image](https://user-images.githubusercontent.com/45097022/113566013-ea6fba80-9646-11eb-9627-450b4d5b3ded.png)
추후 강의에 자세하게 다룰 예정임<br/><br/>

