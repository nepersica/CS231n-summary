# Lecture 14 (Reinforcement Learning)
- 학습일: 2021/05/31
- 주제: Reinforcement Learning

# Overview
- Reinforcement Learning
- Markov Dicision Process (MDP)
- Q-Learning
- Policy Gradients

# Reinforcement Learning
- Agent와 Environment가 존재하며 아래 순서에 따라 진행된다.
    1. Env->Agent에는 State s_t를 전달함.
    2. Agent->Env에 Action a_t를 전달함.
    3. Env->Agent에게 이 Action에 따른 Reward r_t와 Next State s_t+1을 전달함.

## Problem #1 - Cart-Pole Problem
![image](https://user-images.githubusercontent.com/5201073/120140847-59682a80-c216-11eb-902a-38b51212d96a.png)
- 목표: Pole을 움직이는 카트의 수직으로 세우기!

## Problem #2 - Robot Locomotion
![image](https://user-images.githubusercontent.com/5201073/120140861-62f19280-c216-11eb-9b3f-e83d2e597c86.png)
- 목표: Robot이 앞으로 움직이도록 하기!

## Problem #3 - Atari Games
![image](https://user-images.githubusercontent.com/5201073/120140938-8288bb00-c216-11eb-9b82-5cedf2bc187f.png)
- 목표: 게임을 가장 높은 점수로 끝마치기!
- 에이전트가 왼쪽-오른쪽으로 움직일 수 있음

## Problem #4 - Go (바둑)
![image](https://user-images.githubusercontent.com/5201073/120140997-9b916c00-c216-11eb-9b39-0cb881f6afe7.png)

# MDP
![image](https://user-images.githubusercontent.com/5201073/120141086-c24fa280-c216-11eb-8606-538726abb1fb.png)
- Markov Decision Process: 현재 상태만으로 전체 상태를 나타내는 성질임
- S: 가능한 상태들
- A: 가능한 Action들
- R: (state, action) 쌍이 주어졌을때 받는 보상 (보상 Mapping 함수)
- P: 전이확률 (Transition probability)
- gamma: Discount factor (보상을 받는 시간에 대한 중요도 factor)

## How it works?
![image](https://user-images.githubusercontent.com/5201073/120141209-0642a780-c217-11eb-95e6-a8976ce63baa.png)
- 찾으려는 것은 pi^star이다. 이는 최적의 정책을 의미하며 cumulative discounted reward를 최대회시키는 policy이다.

## Example
![image](https://user-images.githubusercontent.com/5201073/120141666-da73f180-c217-11eb-921a-4901d931d817.png)
- 왼쪽-오른쪽-위-아래 움직일 수 있다.
- 한번 움직일때 마다 negative reward를 받는다. -1 이하가 될수도 있을 것이다.

![image](https://user-images.githubusercontent.com/5201073/120141720-f5466600-c217-11eb-9808-184b323f19ee.png)
- 왼쪽 Policy: Random Policy - 어디를 가든 무작위로 간다.
- 오른쪽 Policy: Optimal Policy - 종료 상태에 가장 가깝게 이동할 수 있는 방향으로 이동한다.

![image](https://user-images.githubusercontent.com/5201073/120141789-18711580-c218-11eb-9097-4e03f451301f.png)
- 초기 상태의 무작위성이나 전이 확률 분포의 확률적인 특징을 다 고려하기 위해서 하는 방법?  
→ 보상의 합의 기댓값 E를 최대화하는 pi를 찾으면, 그 값을 이용한 pi가 pi^star이다!

## Value Function & Q-Value Function
![image](https://user-images.githubusercontent.com/5201073/120141907-53734900-c218-11eb-98d8-858b6900202c.png)
- 모든 에피소드마다 결국은 경로를 얻게 됨 (물론 시작은 무작위적이겠지만)
- 상태 s에 대한 Value Function(가치함수)는, 상태 s와 정책 pi가 주어졌을 때의 누적 보상의 기댓값이다.
- Q-Value Function은 어떤 상태 s_i에서 합당한 action a_i를 찾아준다.  
→ 최적의 Q-가치 함수 Q^star는 a_i^star를 찾아줄 것이다.

![image](https://user-images.githubusercontent.com/5201073/120142087-9fbe8900-c218-11eb-93b1-2fde1f7c4a9a.png)
- Q^star가 존재한다고 가정할 때, 그 함수는 Bellman Equation을 만족한다.  
→ 이는, 함수 Q^star의 입력으로 어떤 (s, a)가 주어지던지, 그 결과는  Pair에서 받을 수 있는 reward R과 에피소드가 종료될 s^prime까지의 보상을 더한 값이다.

## Value Iteration Algorithm
![image](https://user-images.githubusercontent.com/5201073/120143709-94209180-c21b-11eb-846c-ed33af0687c8.png)
- Bellman Equation를 이용하여 iterative update를 진행한다.

## Deep Q-Learning
![image](https://user-images.githubusercontent.com/5201073/120144030-2e80d500-c21c-11eb-846a-ed05d88210a0.png)
- Q(s)의 경우 Screen의 모든 Pixel을 상태로 보고 계산해야 하기 때문에 매우 복잡한 계산이 될 것이다!  
→ Deep Q-Learning을 이용하여 이 문제를 해결한다!

![image](https://user-images.githubusercontent.com/5201073/120144064-3ccef100-c21c-11eb-9c8a-9f2f6171c280.png)
- Forward Pass에서는 Bellman Equation 손실 함수를 계산한다.
- Backward Pass에서는 계산한 손실을 기반으로 theta값을 계산한다.

## Example: Q-Network Architecture of Atari Game
![image](https://user-images.githubusercontent.com/5201073/120144195-7b64ab80-c21c-11eb-918a-e3a9e17f30f6.png)
- Last 4 frame을 Stack하여 Conversion, Downsampling등을 진행한다.
- 결과적으로 현재 상태 s_t와, 여기에 존재하는 행동 a_1 ~ a_4까지에 대한 Q 값들이 나오게 된다  
→ 현재 상태에서 취할 수 있는 행동은 왼쪽, 오른쪽, 위, 아래 방향키이기 때문이다!
- 장점: Single Forward Pass만으로 모든 함수에 대한 Q-value(a_1~a_4)를 계산할 수 있다!

![image](https://user-images.githubusercontent.com/5201073/120144537-0c3b8700-c21d-11eb-924f-e5fc1a008094.png)
- Q-Network를 학습하는 데에는 위에서 다룬 Bellman Equation을 Loss Function으로 하여 학습을 진행한다.

## Example: Problems
![image](https://user-images.githubusercontent.com/5201073/120144735-72280e80-c21d-11eb-8a5b-0f88f7ee1811.png)
1. 모든 Sample들이 상관관계를 가짐
2. 현재의 Action 결정이 다음의 샘플을 결정하게 됨  
→ 예를 들어, 현재의 State를 가지고 Action을 결정하면 다음 State가 달라진다! (Bad Feedback Loops)  
→ Replay Memory 방식을 이용하여 문제를 해결할 수 있음!

### Replay Memory Approach
- Replay Memory := [(State_i, Action_i, Reward_i, State_i+1), ...]
- 에피소드를 플레이하면서 더 많은 경험을 얻어가면서 이 테이블을 업데이트한다.
- 이 Memory 안에서 임의의 minibatch를 이용하여 Q-Network를 학습한다.  
→ 연속적인 샘플을 사용하지 않고, 전이 테이블에서 임의로 샘플링된 값들을 이용하는 방식이다.  
→ 왼쪽 버튼만 누르는 편향적인(=이전과 현재 입력간 상관관계가 있는) 플레이를 방지할 수 있다.

## Algorithmm: Deep Q-Learning with Experience Replay (Memory)
![image](https://user-images.githubusercontent.com/5201073/120144997-d2b74b80-c21d-11eb-94c7-aa7e5c621ec6.png)

## Policy Gradients
![image](https://user-images.githubusercontent.com/5201073/120145560-abad4980-c21e-11eb-855d-1191e527b809.png)
- Q-Function이 매우 복잡해지면 (state, action) pair를 찾는 일이 기하급수적으로 어려워진다!  
→ Policy를 directly하게 학습할 수 있다면 Best approach일 것이다!
- Policy의 parameter들을 Gradient **Ascent**로 업데이트하는 방법!  
→ Policy parameter J(theta)의 gradient가 커지는 방향으로 학습하게 된다.

### vanila REINFORCE algorithm
![image](https://user-images.githubusercontent.com/5201073/120145704-ec0cc780-c21e-11eb-9706-2c05c45305ec.png)

![image](https://user-images.githubusercontent.com/5201073/120145720-f16a1200-c21e-11eb-8d24-a93211720419.png)
- J(theta)의 경우 Directly intractable이므로(tau의 적분이 불가능하다), p를 이용하여 식을 변형한다.  
→ 그래디언트를 추정하기 위해 Monte Carlo 샘플링을 이용한다.

![image](https://user-images.githubusercontent.com/5201073/120145957-4e65c800-c21f-11eb-820a-ef97ce0e2868.png)
- 또한, 그래디언트를 추정할 때에는 전이확률 p(tau; theta) 자체를 계산할 필요가 없어진다  
→ 추정시에는 sum of nabla(미분기호)_theta log pi_theta (a_t | s_t) 식을 이용하므로 사실상 확률을 계산할 필요가 없어진다  
→ 확률없이 수식계산만을 이용해서 그래디언트 추정이 가능해졌다! 이는 한번에 여러개의 경로를 샘플링할 수 있다는 의미이다!
- 결국 **경로**에 대한 보상은 그 확률과 같은 맥락을 가진다. 확률 p가 높을수록 행동을 잘했다는 것을 의미하고,  
확률이 낮을수록 그 행동은 좋지 못한 것을 의미한다.
- 문제점: 모든 행동에 대한 확률을 Average out시킨다.(행동에 대한 확률이 아닌 전체 경로로 이루어진 trajectory가 좋은지, 나쁜지를 결정한다.) 이에 따라서 분산이 높아지게 된다...  
→ 보상을 받을 때에는 그 경로가 좋았다는 정보를 줄 뿐이지, 경로 안의 각각 행동들이 모두 좋았다는 것을 의미하는 것이 아니기 때문이다.

### Variance Reduction Technique
![image](https://user-images.githubusercontent.com/5201073/120146910-c7b1ea80-c220-11eb-82a8-1aa5fd2c23b6.png)
- 결국은 Deep Q-Learning with Policy Gradients에서는 분산을 줄이는 과정이 중요하다.

#### Approach #1: 미래의 보상만을 고려한 행동
![image](https://user-images.githubusercontent.com/5201073/120146924-ce406200-c220-11eb-99ef-853d6cc7c70d.png)
- 맨 처음부터의 보상의 합을 구하는 것이 아닌, 현재 시점부터 종료 시점까지의 보상의 합을 고려한다.

#### Approach #2: 지연된 보상에 할인을 적용하기
![image](https://user-images.githubusercontent.com/5201073/120146938-d39dac80-c220-11eb-8fca-2bf27a5cdfe3.png)
- 나중에 수행하는 행동에 대해서는 조금 덜 가중치를 적용한다.

#### Approach #3: 베이스라인
![image](https://user-images.githubusercontent.com/5201073/120146957-db5d5100-c220-11eb-94df-191bd1a02217.png)
- 보상에 기준(baseline)을 정하게 된다.
- 우리가 기대하는 보상값을 정해놓고 (R_U), 미래에 얻을 보상의 합(sum of R_i, i from this_time to end_of_game)에서 이 R_U를 뺀 값을 보상의 합으로 둔다.  
→ 지금 주어진 보상이 다른 trajectory에 비해서 상대적으로 좋은 것인지, 나쁜 것인지를 알려주는 기준을 정해준다.

#### baseline 선정법
1. 경험한 보상들에 대한 Moving Average  
→ 여태까지 경험했던 보상들을 평균내어 Baseline으로 사용한다.  
→ trajectory마다 계속해서 바뀌므로 이를 Moving Average라고 부르는 것

2. 가치 함수 활용법
![image](https://user-images.githubusercontent.com/5201073/120147382-8a019180-c221-11eb-91ac-602d18470225.png)
- 지금까지 해왔던 행동들의 가치 함수 결과보다, 지금의 행동의 가치 함수 결과가 더 큰값을 가진다면?  
→ 이는 지금 한 행동이 이전의 어떤 행동들보다 더 좋았다는 것을 의미한다. (반대면 비교적 좋지 않았다는 것을 의미)

### Actor-Critic Algorithm: Intuitive
![image](https://user-images.githubusercontent.com/5201073/120147564-d1881d80-c221-11eb-9d6c-0c43fbeb2ddf.png)
- 기존에는 Q-function과 Value-function을 구하는 작업을 하지는 않았음  
→ but 이 둘은 Q-learning (Policy Gradient)를 이용하여 학습시킬 수 있다!
- Actor == Policy, Critic == Q-Function이다.  
- 보상함수 R을 재정의한다 -> `Q(s, a) - V(s)`  
→ 행동이 Q-function이 예상했던 것보다 얼마나 더 좋은지를 나타낸다.

#### Actor-Critic Algorithm: Algorithm
![image](https://user-images.githubusercontent.com/5201073/120147835-43606700-c222-11eb-85c0-53d1424133f5.png)

### REINFORCE: Example #1 - Recurrent Attention Model
#### Hard Attention
![image](https://user-images.githubusercontent.com/5201073/120149115-380e3b00-c224-11eb-80e1-66bfe04472a1.png)
- Image Classification을 일부 Glimpses만 가지고 진행한다 (이미지의 빨간 부분)  
→ 이 경우 이미지의 지역적인 부분만 보고 Classification한다.  
(사람의 안구 운동에 따른 지각 능력으로부터 영감을 받음)

- 장점: Computation Resource를 절약할 수 있음 (이미지 전체를 처리할 필요가 없음)  
→ 저해상도를 살펴보면서 어디서부터 시작할지를 정한 뒤, 고해상도에서는 해당 부분에서 세부적인 특징을 찾아내는 방식  
→ 필요없는 부분을 무시할 수 있게 된다.
- 이 방식을 구현하는데 RNN을 이용하게 된다 (Because, Glimpses를 지정하는 Task 자체는 Intractable하기 때문이다.)

#### How it works?
![image](https://user-images.githubusercontent.com/5201073/120150067-8a9c2700-c225-11eb-890f-1dbf662cb53b.png)
- 행동 분포 A = {a_1, a_2, ...} 로부터 Nerual Network를 통과시키면, 다음 Glimpse의 x, y 좌표를 얻어낸다.
- 이 x, y 좌표에 해당하는 위치의 Glimpse (Patch)를 얻어내고, 위 과정을 다시 반복한다.
- 과정을 반복하면서 "현재 상태와 이전 상태를 입력받는 RNN"을 구성하고, 이 구성에서의 Policy를 모델링한다.
- 보통 과정의 갯수는 6~8번 정도이다.
- 마지막 Step에서는 각 클래스의 확률분포를 출력하도록 하고, Softmax를 이용하여 그 결과를 뽑아본다.

#### Applicable Systems
- Image Captioning, VQA(Visual Question Answering), AlphaGo

# Conclusion
![image](https://user-images.githubusercontent.com/5201073/120152733-cc7a9c80-c228-11eb-93f1-d60528bb5fc5.png)
- 충분한 Exploration이 필요할 것이다!