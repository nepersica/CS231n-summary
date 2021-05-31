# Lecture 12 (Visualizing and Understanding)
- 학습일: 2021/05/03 ~ 05/12
- 주제: Detection, Segmentation

# Repraise
1. Semantic Segmentation
2. Classification + Localization
3. Pose Recognition

# Layer Visualization
## First layer visualization
![image](https://user-images.githubusercontent.com/5201073/117605180-1b856280-b192-11eb-9c97-ae3587043218.png)
- Pretrained Model에 대한 첫 레이어에 대한 시각화를 진행해보면 3x11x11에 대한 Visualization이 가능하다.
- 하지만 내부의 필터들은 Channel의 갯수가 3개가 아니므로 Visualize하기 힘들다 + Intuition을 얻기 힘들다.  
→ 이는 첫번째 레이어의 결과를 최대화시키는 첫번째 레이어의 출력 패턴에 따른 것이므로 직관적으로 보기 힘들다.

## Last layer (FC) visualization
![image](https://user-images.githubusercontent.com/5201073/117605511-d57cce80-b192-11eb-83cf-651b60b22616.png)
- 마지막 4096-vector에 대한 Nearest Neighbor가 가까운 이미지들을 나열하는 것이 도음이 된다!
- 위 이미지를 보면, 네트워크가 학습을 통해서 비슷하지만 다른 카테고리의 이미지의 Semantic Content도 알아서 추출한다는 반증을 확인해볼 수 있다.

## PCA (Principle Component Analysis)
![image](https://user-images.githubusercontent.com/5201073/117605638-2c82a380-b193-11eb-9e4a-169a0491813b.png)
- t-SNE(t-distributed Stochastic Neighbor / Embeddings)가 더 강력한 알고리즘이다.
- MNIST의 28x28의 입력을 받아 2-dim으로 압축하여, 이를 이용하여 MNIST 데이터셋을 시각화할 수 있다 (빅데이터 시각화 기법?!)

## Activation visualization
### Visualzation Techniques
![image](https://user-images.githubusercontent.com/5201073/117605868-a024b080-b193-11eb-8b7b-4b1238d27970.png)
    - 중간 레이어의 Feature Map 128x13x13을 128개의 13x13x1 이미지로 표현한다.

### Occulusion Expermient
![image](https://user-images.githubusercontent.com/5201073/117609290-b5e9a400-b19a-11eb-8dad-4b3378109925.png)
- 이미지의 일부분을 가려놓고 데이터셋의 평균으로 채웠을 때, 어떤 부분을 가렸을 때 분류가 많이 달라진다는 의미는 그 부분이 분류 결정에 중요한 역할을 한다는 반증이다.  
→ Sailency Map!

### Gradient Ascent
![image](https://user-images.githubusercontent.com/5201073/117609583-3f997180-b19b-11eb-9d29-3568b5d867fb.png)
- Gradient를 오히려 Ascent시켜서, Neuron의 Activation을 최대화시키는 이미지를 생성하게 된다.
- 여기서도 Regularization Term을 이용한다  
→ 이미지가 자연스러워보이게 하기 위함이다.

### DeepDream
![image](https://user-images.githubusercontent.com/5201073/117613924-8ccd1180-b1a2-11eb-8fda-cf88d4aabecd.JPG)
- 특정 레이어까지 Forward한 뒤에, 그 위치에서 Backpropagation을 진행하는데 이 때의 Gradient를 Activation Value와 동일하게 설정하고 **원본 이미지를 업데이트**한다!   
→ activation == gradient가 되면 결국 원본 이미지는 Activation이 커지는 방향으로 변화하게 된다.
- 이를 Multiscale로 반복하면 위와 같은 이미지를 얻을 수 있다!

### Feature Inversion
![image](https://user-images.githubusercontent.com/5201073/117614068-bede7380-b1a2-11eb-84d8-4e6cfff216cf.png)
- 두가지 이미지를 VGG-16에 통과시켰을때, 중간 Feature를 기록해두었다가 합성하여 이 Feature가 가지고 있는 정보의 양을 확인할 수 있다.  
→ 뒤로 갈 수록 점점 Edge와 같은 Low-level 정보가 적어지고 Color, Texture와 같은 High-level detail들을 확인할 수 있다.

### Neural Texture Synthesis
![image](https://user-images.githubusercontent.com/5201073/117614340-24cafb00-b1a3-11eb-8fda-bc1526d80b1d.png)
- 주어진 패치를 더 크게 만드는 Task
- 보통은 Nearest Neighbor 등 고전적인 방식으로 패치들을 합친다.

![image](https://user-images.githubusercontent.com/5201073/117614469-5217a900-b1a3-11eb-8a22-f2e3d9ee9c65.png)
- 다만 복잡한 텍스쳐는 잘 작동하지 않는다....
- 이 때, 네트워크에 이미지를 통과시켜서 중간의 특정 레이어에서 Feature Map을 가져와서 Descriptor를 계산해서, Gram Matrix를 계산한다.
- 이 Gram-matrix에는 공간정보가 전혀 들어있지 않다. 다만 이 Matrix는 Image texture의 Co-occurance만을 포착해 그 정보를 가지고 있다.

![image](https://user-images.githubusercontent.com/5201073/117615023-1f21e500-b1a4-11eb-868c-c4d6e19fc53f.png)
- 생성할 이미지를 랜덤으로 초기화하고, 만들어진 Gram Matrix를 계산해서 이를 Minimize하는 방식으로 Image를 업데이트한다.  
→ GAN의 전신?

![image](https://user-images.githubusercontent.com/5201073/117614964-09142480-b1a4-11eb-92f2-3c987f262c25.png)
- 공간적인 정보가 날라가기 3, 4번째와 같은 일반적인 이미지에서는 잘 동작하지 않는 것을 확인할 수 있다.

# Nerual Style Transfer
![image](https://user-images.githubusercontent.com/5201073/117615283-79bb4100-b1a4-11eb-8794-3a91d0ced3fe.png)
- 원본 이미지를 Style Image풍으로 만들 수 있는 Task이다!
- 단점은 매우 느린 Task라는 것! 4K 이미지 하나 만드는데 GPU 4대가 쓰인다...  
→ Forward/Backward pass를 많이 해야하기 때문이다.

## Fast Style Transfer
![image](https://user-images.githubusercontent.com/5201073/117907568-fe779d80-b311-11eb-85ac-084acbbe8674.png)
- Neural Style Transfer의 느린 스피드를 해결하기 위해서 등장했다.  
→ 단일 Forward/Backward Pass만으로 이미지를 생성할 수 있도록 한다!
- 몇천배 빠르게 동작함
- Single Forward-pass만 해도 결과가 잘 나온다! 