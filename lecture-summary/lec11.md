# Lecture 11 (Detection and Segmentation)
- 학습일: 2021/05/03 ~ 05/10
- 주제: Detection, Segmentation

# Repraise
- RNN을 이용한 텍스트 학습
- RNN의 문제점을 보완한 기법 소개 (Gradient Cliping, LSTM)
- LSTM 구조 및 작동 원리 소개

# 01 Semantic Segmentation Task
- 입력: 이미지 전체, 출력: 이미지 크기에 해당하는 모든 픽셀의 카테고리  
→ 예를 들어, 고양이 이미지가 들어오면 하늘 픽셀들, 고양이 픽셀 등을 분류한다.
- 개별 객체를 구분하지 않는 방법이다.

## 접근법
1. Classification Problem
    - Sliding Window를 적용해볼 수 있다.  
    → 이미지를 아주 작은 단위로 쪼개서 이 단위들을 Classification한다.  
    → 문제는 비용이 엄청 크다는 것!
    - Fully-Convoltional Network를 이용한다.  
    → FC 레이어가 없이 Conv만으로 네트워크를 구성한다.  
    → 모든 픽셀의 Classification Loss를 계산한뒤 평균을 취한다.  
    그리고 기존 방식과 동일하게 Backpropagation을 수행한다.
    - 고질적인 문제: 엄청 큰 메모리 사용량  
    → 전체 이미지를 가지고 계산하기에는 메모리 사용량이 기하급수적으로 많아진다.  
    이를 해결하기 위해서 보통 이미지를 Low-res로 Downsampling(Squeeze)한 뒤에, 이 정보들을 가지고 다시 High-res로 Upsampling(Expand)하는 과정을 거치게 된다.  
    이러한 방법을 사용하면 Gradient 전달을 할 수 있게 할 뿐만 아니라 메모리 사용량을 줄이는(Spatial size를 줄이는) 효과를 얻게 된다.

### Upsampling
![image](https://user-images.githubusercontent.com/5201073/116845740-abbb2900-ac21-11eb-9c2a-9e5aa53147b8.png)
    - Downsample된 값들을 다시 Upsample하는 과정
    - 한 부분만 뾰족하게 튀어나와있다는 특징이 있음

![image](https://user-images.githubusercontent.com/5201073/116847470-8e885980-ac25-11eb-9546-4c66ebdf660e.png)
    - Unpooling 수행 시 데이터가 원래 있던 자리에 값을 다시 붙여놓음
    - 이는 원래 값이 있던 위치에 그 값을 되돌려놓는다는 점에서 Spatial information을 보존한다는 의의가 있음

### Transposed Convolution
![image](https://user-images.githubusercontent.com/5201073/116847599-d27b5e80-ac25-11eb-9fca-883dd6a18226.png)
    - 파란색과 빨간색 Box가 겹치는 공간은 간단히 더한다.
    - 달리 불리는 이름들 (1번 제외!)  
        ~~1. Deconvolution~~  
        2. Upconvolution  
        3. Fractically Strided Convolution  
        4. Backward Strided Convolution

# 02 Classification + Localization Task
- Object Detection과는 구별되는 문제임  
→ Localization 문제에서는 이미지에 관심 Object가 단 하나만 있으나, Classification+Localization Task에서는 여러 개의 관심 객체에 대한 Detection을 수행하게 됨.
- Classification과 Box Regression Task를 동시에 수행함 (Multi-task Loss)

# 03 Human Pose Estimation Task
- 사람의 경우 14개의 고정된 갯수의 관절(Joints)을 가지고 있다.
- 이 관절 14개에 대한 각 좌표들을 Regression Loss를 계산하여 학습하는 Task이다. 

# 04 Object Detection
- Classification + Localization Task에서는 지정된 갯수의 Bounding Box를 찾았지만, Object Detection에서는 n개의 Bounding Box를 찾는 Task이다.

## Finding bounding box (without class)
### 01 Brute-Force, Sliding Window
- 매우 많은 이미지 Patch들을 네트워크에 통과시켜야하므로 매우 느리다.

### 02 Region Proposals
- CPU로 수행하며, 신호처리 알고리즘을 이용하여 Object가 있을만한 위치를 추천(Propose)한다.
- Selective Search: Region Proposal의 기법 중 하나이며, Recall은 아주 높지만 Noise에 민감하다.

#### RPN/Algorithm-based BBOX Prediction
- R-CNN(49s), SPP-Net(4.3s), Fast R-CNN(2.3s), Faster R-CNN(0.2s)

#### CNN-based BBOX Prediction
- YOLO, SSD 등등

# 05 Instance Segmentation
- Semantic Segmentation과 Object Detection을 합친 분야임
- 여러 객체를 찾고 각각을 구분하는 문제임
- Mask-R-CNN