# Lecture 15 (Efficient Methods and Hardware for Deep Learning)
![image](https://user-images.githubusercontent.com/5201073/120153413-a73a5e00-c229-11eb-8091-eb692082e9d3.png)
- 학습일: 2021/05/31
- 주제: Efficient Methods and HWs - Algorithms/Hardware, Inference/Training

# Prior Problems
- Large model takes large storage (and bandwidth to update)
- and also large model requires lot of memory and energy!

# Part #1 - Algorithms for Efficient Inference
1. Pruning (~90% is possible in some cases)
2. Weight Sharing
3. Quantization
    - DataType Accuracy Quantization (INT8, FP16, etc...)
    - Huffman Coding
    - from "LeNet-300 40x compression" to "ResNet-18 11x compression"...
    - SqueezeNet (510x compression)
    - Training with FP32 accuracy + Quantizing weight and activations
4. Low-Rank Approximation
    - Break one FC layer into lots of FC layers!
5. Binary / Ternary Net
    - Networks using 0, 1 (binary) or -1, 0, 1 (ternary)
6. Winograd Transformation
    - "Fast Algorithms for Convolutional Neural Networks", known as Winograd, NVIDIA

# Part #2 - Algorithms for Efficient Training
1. Parallelization
    - Data Parallel + Parameter Update (like PyTorch's DataParallel Approach)
    - Model Parallel
        ![image](https://user-images.githubusercontent.com/5201073/120156249-c4246080-c22c-11eb-8ab0-f8daf1308902.png)
        - Cut model in half to parallelizate model training
2. Mixed Precision with FP16 and FP32
    ![image](https://user-images.githubusercontent.com/5201073/120156455-f46bff00-c22c-11eb-90b0-07781a83425f.png)
3. Model Distillation
    - Example) GoogleNet+VGGNet+ResNet (Teacher Model) => OneModel (Student Model)
    ![image](https://user-images.githubusercontent.com/5201073/120156762-41e86c00-c22d-11eb-9aba-dc258ee2f351.png)
4. DSD: Dense-Sparse-Dense Training
    ![image](https://user-images.githubusercontent.com/5201073/120156840-5a588680-c22d-11eb-832b-cc3091857098.png)
    - DSD Model Zoo: https://songhan.github.io/DSD/
