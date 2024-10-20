# Neural-Networks from Scratch - An Exercise

This project is an exercise to build various neural-networks from scratch. 
Of course, professional libraries such as PyTorch or TensorFlow are far better optimized,
however the idea of this project is not optimization but rather the understanding of the inner workings.
As toady's models have become so big that it seems notoriously difficult to understand their decision making exactly, 
I think that it is a good idea to return to the fundamentals, especially when learning the field.

My former supervisor and friend often claims that the only theorem they truly understand is that 1 < 2.

This project is in their spirit: Firstly, small exercises aimed at understanding neural networks of the simplest possible architecture. 
Afterwards, we can focus on 2.

## Contents

* **xor.py** - script solving xor. This script serves as a quick test for the implementation of the dense layer.

* **mnist1d.py**, **mnist2d.py** - scripts solving mnist's image classification. The scripts serve as tests for the 1d and 2d CNN layer. mnist1d.py classifies for simplicity only images of 0 and 1 using a final sigmoid activation and a binary_cross_entropy loss; mnist2d.py classifies images of 0,1 and 2 using a final softmax activation and a categorical cross entropy loss (it hence serves equally as a test for the softmax layer and the CCE loss)
