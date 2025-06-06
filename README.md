# AI Independent Project

Final Goal: build a CNN with Numpy to classify MNIST dataset.

## Reference Sources

- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)
- [CNN with NumPy](https://github.com/SkalskiP/ILearnDeepLearning.py/tree/master/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net), a reference codebase.

## Tentative Schedule

- [X] Jun 3: 
    - reading (30 minutes): the Stanford CS231n website, focusing on the concept of convolution, padding, stride, and pooling.
    - coding (30 minutes): load the MNIST and CIFAR-10 datasets. Design the class structure for CNN, including Convolution, MaxPooling, FullyConnected, and Softmax layers.

- [X] Jun 4:
    - reading (30 minutes): the source code from the reference codebase, focusing on the implementation of convolutional layers and pooling layers.
    - coding (60 minutes): 
        - implement the forward pass for the convolutional layer and pooling layer, and do tests to ensure correctness.
        - build a neural network using the implemented layers, also including ReLU, flattening, and softmax layers.

- [X] Jun 5:
    - reading (30 minutes): the source code from the reference codebase, focusing on the implementation of backpropagation for convolutional layers and pooling layers.
    - coding (60 minutes): 
        - implement the backward pass for the convolutional layer and pooling layer, and other layers
        - run backpropagation for several epoch on the MNIST dataset, and check if the loss is decreasing.
        - test the accuracy of the model on the MNIST dataset.