# AI Independent Project

Final Goal: build a CNN with Numpy to classify MNIST and CIFAR-10 datasets.

## Reference Sources

- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)
- [CNN with NumPy](https://github.com/SkalskiP/ILearnDeepLearning.py/tree/master/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net), a reference codebase.

## Tentative Schedule

- [X] Jun 3: 
    - reading (30 minutes): the Stanford CS231n website, focusing on the concept of convolution, padding, stride, and pooling.
    - coding (30 minutes): load the MNIST and CIFAR-10 datasets. Design the class structure for CNN, including Convolution, MaxPooling, FullyConnected, and Softmax layers.

- [ ] Jun 4:
    - reading (15 minutes): the source code from the reference codebase, focusing on the implementation of convolutional layers and pooling layers.
    - coding (45 minutes): implement the forward pass for the neural network, and sanity check the output shape.

- [ ] Jun 5:
    - reading (15 minutes): the source code from the reference codebase, focusing on the backward pass and gradient computation.
    - coding (45 minutes): implement the backward pass for the neural network, and sanity check that loss can decrease.

- [ ] Jun 6:
    - experimenting (30 minutes): train the CNN on MNIST and CIFAR-10 datasets, and observe the training and validation accuracy.
    - summary (30 minutes): document the results.