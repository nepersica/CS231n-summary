# Lecture 13 (Generative Models)
- 학습일: 2021/05/12
- 주제: GANs (Generative Mdoel)

# Unsupervised Learning - Generative Model
- 비지도 학습: 레이블이 없는 상태에서 데이터를 학습시켜야 함!  
→ 데이터의 기본적인 구조를 클러스터링(군집화)하여 나눠야 한다
    1. K-means Clustering
    2. Dimension Reduction (차원 축소)
    3. Feature Representation Training
    4. Autoencoders
- 비교

    | Supervised Learning                         | Unsupervised Learning                                   |
    |---------------------------------------------|---------------------------------------------------------|
    | Data: (x, y)                                | Data: (x)                                               |
    | Goal: Learn function function(x) -> y       | Goal: Learn hidden structure of data                    |
    | Examples: Classification ~ Image Captioning | Examples: Clustering, Dim Reduce, Feature Learning, etc |

- 장점
    - 데이터를 추가하는데 비용이 거의 들지 않음 (레이블이 필요없음)

- 활용가능성
![image](https://user-images.githubusercontent.com/5201073/119090971-4c458180-ba47-11eb-9d87-0da6fef8e2d4.png)
    - Super Resolution
    - Colorization
    - Planning

## pixelRNN
- Fully-visible belief network
- Explicit density model
![image](https://user-images.githubusercontent.com/5201073/119091065-76973f00-ba47-11eb-9ff5-abaf7ea2173c.png)
- 우도 `p(x)`를 정의하고 이를 최대화시키는 방향으로 모델을 학습한다.  
→ Neural Network를 이용하여 이를 가능하게 한다.
![image](https://user-images.githubusercontent.com/5201073/119091282-c70e9c80-ba47-11eb-8760-f5c168c7fb7d.png)
- RNN을 이용하여 왼쪽 상단부터 이미지를 생성하게 된다  
→ 여기서 좌상단 픽셀의 값이 매우 중요하게 된다.

## pixelCNN
- pixelRNN과 동일하게 왼쪽 코너부터 이미지를 생성함
- RNN이 아닌 CNN을 이용하염 모델링 진행
- 학습 자체는 빠를 수 있으나 이미지 생성에는 pixelRNN과 동일하게 오랜 시간이 걸린다.
- Audio Generation에서도 사용되는 기법이다.


## Variational Autoencoder (VAE)
![image](https://user-images.githubusercontent.com/5201073/119096162-31c2d680-ba4e-11eb-99b5-fb9a024acde3.png)
- 직접 계산이 불가능한 확률 모델(intractable model)을 정의한다.  
→ 그리고 이 모델을 최적화하는 방법으로 모델을 학습시킨다.
- Autoencoder는 데이터 생성이 1차 목적이 아님  
→ 데이터로부터 어떤 데이터든 잘 표현할 수 있는 Latent feature vector를 구하는 것이 목표이다!  
→ (Feature representation의 학습)
- 사실 데이터를 생성하는 목적은 아니고, 많은 데이터를 이용하여 저차원의 Feature Representation을 학습하기 위한 준-지도 방법임
(Unsupervised Approach, not an Unsupervised Learning)
- Encoder-Decoder 구조이며, Decoder는 이미지를 Feature Vector로 디코딩(임베딩)하는 역할을,  
Encoder는 이 벡터를 가지고 다시 이미지를 생성하는 역할을 한다.

### Activations
![image](https://user-images.githubusercontent.com/5201073/120129093-57926d00-c1fe-11eb-9860-7019bac69587.png)
- Input Data와 Reconstructed Data 사이에는 L2 Loss를 사용함
- Decoder는 사용한 뒤에 버림

### Structure
- Encoder == recognition(inference) network
- Decoder == generation network

### Training
- 잠재 변수 z를 만들기위한 Encoder-Decoder 구조를 학습해야 함
- `p(x)`를 쉽게 모델링할 수 있는 Gaussian Distribution으로 정의한다.
- likelihood `p(x)`를 최대화하는 방법은?  
→ 적분 계산을 할 수는 없음. 
![image](https://user-images.githubusercontent.com/5201073/120137556-90870d80-c20f-11eb-87e1-018a9e4454f2.png)
- 위의 이미지에서와 같이 Gaussian 분포 `p(x)`를 최소화시키기 위해 KL-Divergence를 이용한다. 이 KL-Divergence를 최소화하는 문제를 정의하고 이를 위한 학습을 진행하게 된다.
- 결국 위에서 원하는 결과는 latent variable z가 prior distribution(gaussian distribution)과 유사하도록 유도한다.

### Training: Details
![image](https://user-images.githubusercontent.com/5201073/120137740-e956a600-c20f-11eb-8842-ff914a92c2e0.png)
- 먼저 Input Data를 Encoder Network q_theta(z|x)에 통과시킨다.

![image](https://user-images.githubusercontent.com/5201073/120137777-02f7ed80-c210-11eb-9644-3b33dc6a0519.png)
- 다음으로는 위 과정의 출력 z의 평균과 분산을 Decoder Network p_theta(x|z)에 통과시킨다.  
→ Decoder Network의 출력으로는 hat x가 출력된다.

![image](https://user-images.githubusercontent.com/5201073/120137862-36d31300-c210-11eb-983e-4362ef6b43a4.png)
- 디코더 네트워크를 거친 hat x와 입력 이미지 x를 이용해서, logp(input_image | z)를 최대가 되도록 학습하게 된다. 슬라이드에서는 L(x^i, theta, pi)라고 표시되었다.

![image](https://user-images.githubusercontent.com/5201073/120138023-8d405180-c210-11eb-9219-31b9a3915b8b.png)
- 결과적으로 MNIST 데이터셋을 학습했을 때에, 2차원 latent vector z의 값에 따라 위와 같은 결과 Data Manifold를 만들어내는 것을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/5201073/120138079-b365f180-c210-11eb-840c-bb750faff9ef.png)
- 표정을 생성해내는데에도 사용할 수 있다!

### Problems
![image](https://user-images.githubusercontent.com/5201073/120138138-dabcbe80-c210-11eb-9a7b-92d5c5b55c98.png)
- Blurry Image를 생성해낸다.

### Conclusion
![image](https://user-images.githubusercontent.com/5201073/120138168-ec9e6180-c210-11eb-9100-f195acd503eb.png)
- 계산할 수 없는(Intractable) 분포를 다루기 위해서 Variational Lower Bound라는 개념을 도입하였다  
→ 계산할 수 없는 형태를 위한 **근사 기법**이다.  
→ 정확히는 p(z|x)를 계산할 수 없으니, q(z|x)로 근사하는 것)

# GAN (Generative Adversarial Networks)
![image](https://user-images.githubusercontent.com/5201073/120138511-941b9400-c211-11eb-9091-e0e6d6c26194.png)
- GAN에서는 Gaussian과 같은 분포를 직접 지정하지 않고, 게임 이론에 따라 이 분포를 스스로 학습하도록 한다.

- Generator: Random Noise로부터 이미지를 생성해낸다.
- Discriminator: 이미지가 진짜인지 가짜인지 구분한다.

![image](https://user-images.githubusercontent.com/5201073/120139481-86670e00-c213-11eb-8a19-15485891ff6b.png)
- D(x)와 G(x)를 각각 학습시키는 방식으로 모델을 학습시킨다.
- Random Noise가 Generator G(x)에 입력되면, 이 값을 이용하여 Generated Image를 만들어낸다.  
→ Discriminator D(x)는 이미지를 입력받아 이미지가 진짜인지, 거짓인지를 확인해낸다.

![image](https://user-images.githubusercontent.com/5201073/120139481-86670e00-c213-11eb-8a19-15485891ff6b.png)
- Generator를 학습하기 위해서 사용하는 목적 함수의 경우, Bad sample에 대해 더 큰 가중치를 두기 위해 2번 수식의 목적 함수를 사용한다!

## Training
![image](https://user-images.githubusercontent.com/5201073/120140141-dc888100-c214-11eb-880e-dcfdcea1e54a.png)
- Generator와 Discriminator를 각각 학습시키는 방식을 이용해야 한다.

## After vanila GAN (2017년 기준!!)
![image](https://user-images.githubusercontent.com/5201073/120140275-28d3c100-c215-11eb-9900-555961837bd4.png)
- LSGAN, 2017
- BEGAN, 2017
- CycleGAN, 2017
- Text-to-Image Synthesis (Akata et al. 2017)
- Pix2pix

## Conclusion
- 특정 확률분포를 정의(Explicit density)하여 KL-divergence를 이용하지 않고, Implicit한 방법을 사용하였다.  
→ Two Player Game을 이용하여 학습 데이터의 분포로부터 생성 모델을 정의해내었다.
- 단점: 학습시키기 까다로움 (Generator의 경우) + Objective function을 직접 최적화하는 Task가 아님 (G+D의 Joint 학습임)  
→ + VAE와 같이 Inference Query가 불가능함! (p(x)나 p(z|x)와 같은 Querying)