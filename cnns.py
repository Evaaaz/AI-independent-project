import numpy as np

class Layer:
    def __init__(self):
        pass
    
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, output_gradient):
        raise NotImplementedError

class Convolution(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        raise NotImplementedError
    
    # TODO: implement forward and backward

class Pooling(Layer):
    def __init__(self, kernel_size, stride):
        raise NotImplementedError
    
    # TODO: implement forward and backward
    
class ReLU(Layer):
    pass    

class Flatten(Layer):
    pass

class FullyConnected(Layer):
    def __init__(self, in_features, out_features):
        raise NotImplementedError
    
    # TODO: implement forward and backward

class Softmax(Layer):
    pass

class Sequential(Layer):
    pass


def build_CNN_mnist():
    raise NotImplementedError

def build_CNN_cifar10():
    raise NotImplementedError