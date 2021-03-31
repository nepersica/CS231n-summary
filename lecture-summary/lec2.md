# Lecture 2 (Image Classification)
- 학습일: 2021/03/31
- 주제: 머신러닝 학습 알고리즘

## Image Classification
- 이미지를 보고 어떤 클래스인지 선택하는 문제
![image](https://user-images.githubusercontent.com/5201073/113081648-0a047e80-9214-11eb-88e4-ce19cdb2fe9a.png)
- 컴퓨터가 인식하는 이미지는 R/G/B로 이루어진 값들의 집합이므로 이를 보고 어떤 이미지인지 분류하는 것은 어려운 일임  
++ 카메라를 조금만 움직여도 전체 픽셀이 움직인다!
- Deformation, Occlusion, Background Clutter, Intraclass variance 등  
화각, 객체 자체의 모양과 형태가 변하는 경우가 많아 모델은 이들을 문제 없이 분류할 수 있도록 강인해야 한다.  
→ 사람은 수백 ms만에 이를 수행한다!
- 과거에는 사람이 이를 컴퓨터 알고리즘 함수로 작성하려는 시도를 해왔다.
![image](https://user-images.githubusercontent.com/5201073/113081966-a29afe80-9214-11eb-9e2f-e271d32104bd.png)
- Approach #1: 초기에는 이미지의 Edge를 계산하여 이를 Feature로 하여 이미지를 분류하였다.  
→ 고양이가 움직이거나 다른 자세를 취하면 감지하지 못한다는 큰 문제가 있다.
- Approach #2: Data-Driven Approach, 데이터셋을 많이 수집한 뒤에 Machine Learning Classifier에 학습시킨다.  
→ 지금까지도 계속해서 사용되는 기법이다. 데이터셋과 알고리즘(=모델)을 학습시키는 데에 초점을 맞춘다.

## Data-Driven Approach - Classifiers and Distnace metrics

### 1. Nearest Neighbor Classifier: 모든 저장된 Train set들에 대해 픽셀 거리를 계산한다.
![image](https://user-images.githubusercontent.com/5201073/113082356-45ec1380-9215-11eb-9f61-f5a686b789dd.png)
→ 색깔이 비슷한데 아예 다른 클래스의 이미지가 선택되었다(=거리가 가깝다)는 문제가 있다.  
→ "어떤 비교 방식을 사용할까?"라는 문제가 제기된다.
  - L1 distance
  ![image](https://user-images.githubusercontent.com/5201073/113082484-864b9180-9215-11eb-8819-e2ad73bc95a5.png)
  - Decision boundaries
  ![image](https://user-images.githubusercontent.com/5201073/113082731-f823db00-9215-11eb-90fb-260470d77d69.png)  
  → 공간을 나눠서 레이블을 분류하게 된다. 노란색 점 한개가 사실은 초록색 영역이어야 해서, 이를 정규화할 방법일 필요하다.

### 2. K-Nearest Neighbors
![image](https://user-images.githubusercontent.com/5201073/113082851-2a353d00-9216-11eb-9c32-9b7d74cb2577.png)
→ K개의 최접점의 투표로 최다득표 클래스로 결정한다.
  - L2 distance: Squared sum of powers
  ![image](https://user-images.githubusercontent.com/5201073/113083295-ee4ea780-9216-11eb-9884-5b4b8b0335ed.png)
  - Decision boundaries
  ![image](https://user-images.githubusercontent.com/5201073/113083609-72a12a80-9217-11eb-952d-f63fdeee25b0.png)  
  → Distance metric을 변경함으로써 Distance boundary가 변경된다.

### Hyperparameters
- 어떤 `k` 값을 사용할지, 어떤 `distance metric`을 사용할지에 따라 목표/결과가 달라진다.
- 이러한 값들을 `Hyperparameter`라고 한다.
- *질문: 어떠한 distance를 사용하는 것이 맞을까? → 각 column이 특별한 의미를 가지고 있을 때는 L1 Distance를 사용한다.*

1. 데이터셋 나누기
  ![image](https://user-images.githubusercontent.com/5201073/113083930-12f74f00-9218-11eb-9cd6-5101e2812608.png)  
  → 기존 Dataset으로부터 Validation과 Test를 나누는 것은 매우 중요한 Task이다!
  - 데이터셋을 K개의 Fold로 나누어 학습하는 방식도 사용한다. (Assignment 내 `knn.ipynb` 참조)

### 3. k-NN의 문제점
  1. 이미지 거리 계산 문제
  ![image](https://user-images.githubusercontent.com/5201073/113084536-20610900-9219-11eb-811f-9cad7e93a85b.png)  
  → 같지 않은 Image가 Lower L1/L2 Distance를 가지는 것은 확실한 문제이다.

  2. 차원의 저주(Curse of dimensionality) 문제
  ![image](https://user-images.githubusercontent.com/5201073/113084635-57371f00-9219-11eb-8057-1723619040db.png)  
  → 이미지의 크기가 클 수록, 이미지의 갯수가 (거의)무한으로 많아져야만 한다.  
  (예를 들어, 200x200x3의 이미지로 이루어진 데이터셋은 2^8 * 120,000개의 이미지를 가지고 있어야 공간을 꽉 채워 정확도를 달성할 수 있다!!)

### 4. Linear Classifier
- Parametric Approach
  ![image](https://user-images.githubusercontent.com/5201073/113084885-da587500-9219-11eb-8179-b9ba7609476d.png)
  → 가장 간단한 게산법은 `f(x,W) = W * x + b`이다. (`W = (3072, 1)`, `x = (10, 3072)`, `b = (10, 1)`)
- Linear Classifier 구조
  ![image](https://user-images.githubusercontent.com/5201073/113085249-72565e80-921a-11eb-9949-5bf22bc89d74.png)  
  → Bias는 해당하는 클래스에 무조건적인 순위(편향)를 주는 term이다.
- 고차원 해석
  - Linear Classifier는 고차원 (`200x200x3` 차원)에서는 하나의 점으로 존재한다.  
  → 이를 생각해보면 Linear classification처럼 하나의 템플릿을 가지는 단순한 모델로 이미지를 분류할 수 없다. (아래)
  ![image](https://user-images.githubusercontent.com/5201073/113085592-1d671800-921b-11eb-8e30-8329eda4b6d9.png)
  - 분류기는 검은 선 두 가지를 가지고 위와 같은 상황을 모두 구별할 수 없다는 문제가 있다.