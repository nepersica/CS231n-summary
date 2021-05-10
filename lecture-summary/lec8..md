# Lecture 8 (Depp Learning Software)
- 학습일: 2021/04/19
- 주제: Deep Learning Software

## CPU and GPU
- CPU는 적은 코어수 대비 높은 클럭을 가지고 범용 연산을 수행함
- GPU는 많은 코어수 대비 낮은 클럭을 가지고 Vectorized 또는 Matrix Multiplication 연산을 수행함

## Deep learning framework
### Points
1. Computational Graph 생성이 가능함
2. Gradient를 구하기가 쉬움
3. GPU에서 효율적으로 작동함

### Tensorflow V1에 대한 설명
![image](https://user-images.githubusercontent.com/5201073/115171349-d07daf80-a0fd-11eb-9acf-6044e1354258.png)
    - **모든 Graph 생성 함수들은 In-place가 아니다!**  
    → Graph가 실제로 만들어져 실행되기 전까지는 실질적인 크기의 메모리 할당이 일어나지 않는다. 메타데이터만 구성하며 실제 할당과 연산은 sess.run()에서 수행된다.  
    - 입력 변수를 `tf.placeholder`로 설정하여 Graph를 구성한다.
    - 이 때 메모리 할당은 일어나지 않는다. 오직 Graph만 구성한다.  
    → 아무런 연산도 수행하지 않는다!  
    ![image](https://user-images.githubusercontent.com/5201073/115171911-1a1aca00-a0ff-11eb-993d-2d1343126eef.png)
    - 매번 `sess.run()`을 수행할 때마다 GPU로 복사가 수행된다  
    → `tf.placeholder`가 아닌 `tf.Variable`로 Graph에 Weight를 묶어두자!  
    → 이렇게 하면 `sess.run()`을 수행할 때마다 Weight가 계속 복사되지 않는다!
    ![image](https://user-images.githubusercontent.com/5201073/115172087-70880880-a0ff-11eb-9727-e48b50a906b2.png)
    - 그리고 위와 같이 Weight에 직접 Update시켜준다 (이는 마찬가지로 Graph 내=GPU 내에서 실행된다)

### Difference between two libraries
- Tensorflow는 Graph Build를 미리 해 두고, iteration마다 계속 이를 실행하는 방식이다.
- 반면에, PyTorch는 매 iteration마다 그래프를 생성하는 Dynamic Computational Graph 방식이다.
- Tensorflow에서 조건부 연산을 달성하기 위해서는 별도의 그래프가 만들어져야 한다  
→ 이를 위해 `tf.cond()`를 이용하여 조건부 연산을 명시적으로 정의해야만 한다.
    ![image](https://user-images.githubusercontent.com/5201073/115173273-e5f4d880-a101-11eb-95f6-398e9d7664a0.png)
- 반복문(Loop)을 만들기 위해서 Control Flow를 이용해야 한다...!
    ![image](https://user-images.githubusercontent.com/5201073/115173350-18063a80-a102-11eb-8d58-9b85a1ebc119.png)
- 자연어처리와 같이 non-fixed length graph가 필요한 경우 Dynaimc Graph(→ PyTorch에서 지원하는!)가 적합하다.
    ![image](https://user-images.githubusercontent.com/5201073/115173716-dc1fa500-a102-11eb-9c7d-3002808b9e19.png)
- Dynamic Graph를 그때그때 구축해주는 아이디어 (NeuroModule)임!
- 질문을 하면 그때그때 그래프를 구축하여 문제를 해결함 (maybe RNN-related stuffs...)

### Caffe/Caffe2
- 모델은 prototxt라는 파일에 텍스트로 작성한다.
- 데이터셋은 HDF5나 LMDB의 포맷으로 입력받는다.
- Optimizer나 Solver를 또다른 prototxt 파일에 정의한다...
- Pretrained Model을 제공하여 이들을 활용할 수 있다.
- (지금은 PyTorch에 병합되어버렸다.)
