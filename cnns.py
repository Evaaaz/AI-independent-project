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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.kernel_weight = np.empty((out_channels, in_channels, kernel_size, kernel_size))
        self.kernel_bias = np.empty((out_channels,))
        
    def initialize(self):
        self.kernel_weight = np.random.normal(size=self.kernel_weight.shape) * 0.1
        self.kernel_bias = np.random.normal(size=self.kernel_bias.shape) * 0.1
        
    def forward(self, input):
        batch_size, in_channels, height, width = input.shape
        
        # first, perform padding of the input
        if self.padding > 0:
            padded_input = np.zeros((batch_size, in_channels, height + 2 * self.padding, width + 2 * self.padding))
            padded_input[:, :, self.padding:-self.padding, self.padding:-self.padding] = input
            input = padded_input
            height += 2 * self.padding
            width += 2 * self.padding
            
        # calculate output dimensions
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        for out_x in range(out_height):
            for out_y in range(out_width):
                for kernel_x in range(self.kernel_size):
                    for kernel_y in range(self.kernel_size):
                        output[:, :, out_x, out_y] += np.sum(
                            input[:, None, :, out_x * self.stride + kernel_x, out_y * self.stride + kernel_y] * self.kernel_weight[None, :, :, kernel_x, kernel_y]
                        , axis=2)
                        # in this line:
                        # input: batch, (new axis), in channels
                        # kernel: (new axis), out channels, in channels
                output[:, :, out_x, out_y] += self.kernel_bias[None, :]
        
        return output

class MaxPooling(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, input):
        batch_size, in_channels, height, width = input.shape
        
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        output = np.empty((batch_size, in_channels, out_height, out_width))
        
        for out_x in range(out_height):
            for out_y in range(out_width):
                output[:, :, out_x, out_y] = np.max(
                    input[:, :, out_x * self.stride : out_x * self.stride + self.kernel_size, out_y * self.stride : out_y * self.stride + self.kernel_size],
                    axis=(2, 3)
                )
        
        return output
    
class ReLU(Layer):
    def forward(self, input):
        return np.maximum(0, input)

class Flatten(Layer):
    def forward(self, input):
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)

class FullyConnected(Layer):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = np.empty((in_features, out_features))
        self.bias = np.empty((out_features,))
        
    def initialize(self):
        self.weight = np.random.normal(size=self.weight.shape) * 0.1
        self.bias = np.random.normal(size=self.bias.shape) * 0.1
        
    def forward(self, input):
        return input @ self.weight + self.bias[None, :]

class Softmax(Layer):
    def forward(self, input):
        # numerical trick: subtract the max
        input = input - np.max(input, axis=1, keepdims=True)
        return np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)

class Sequential(Layer):
    def __init__(self, *layers):
        self.layers = layers
        
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def initialize(self):
        for layer in self.layers:
            if hasattr(layer, 'initialize'):
                layer.initialize()

def build_CNN_mnist():
    return Sequential(
        # first convolutional layer
        Convolution(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
        ReLU(),
        MaxPooling(kernel_size=2, stride=2),
        # second convolutional layer
        Convolution(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
        ReLU(),
        MaxPooling(kernel_size=2, stride=2),
        
        Flatten(),
        # fully connected layers
        FullyConnected(in_features=32 * 7 * 7, out_features=128),
        ReLU(),
        FullyConnected(in_features=128, out_features=10),
        
        Softmax()
    )

def build_CNN_cifar10():
    raise NotImplementedError

if __name__ == "__main__":
    print("Testing Convolution Layer")
    
    ##### perform test of the shape for convolution layer
    conv = Convolution(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
    conv.initialize()
    input_data = np.random.randn(2, 3, 5, 5)  # batch size of 2, 3 channels, 5x5 image
    output_data = conv.forward(input_data)
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape) # expected: (2, 2, 5, 5)
    
    
    ##### perform test for the value, when the channel is all 1
    conv = Convolution(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
    conv.initialize()
    input_data = np.ones((1, 1, 5, 5))  # batch size of 1, 1 channel, 5x5 image
    output_data = conv.forward(input_data)
    
    from scipy.signal import correlate2d
    W = conv.kernel_weight[0, 0]  # get the kernel weight for the single channel
    b = conv.kernel_bias[0]  # get the bias for the single channel
    
    scipy_out = correlate2d(input_data[0, 0], W, mode='same') + b
    print('Max error:', np.max(np.abs(output_data[0, 0] - scipy_out)))
    
    
    print("Testing MaxPooling Layer")
    ##### test max pooling
    maxpool = MaxPooling(kernel_size=2, stride=2)
    input_data = np.random.randn(2, 3, 4, 4)  # batch size of 2, 3 channels, 4x4 image
    output_data = maxpool.forward(input_data)
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)  # expected: (2, 3, 2, 2)
    N, C = 2, 3

    ref_out = np.empty((N, C, 2, 2))
    for n in range(N):
        for c in range(C):
            ref_out[n, c, 0, 0] = np.max(input_data[n, c, 0:2, 0:2])
            ref_out[n, c, 0, 1] = np.max(input_data[n, c, 0:2, 2:4])
            ref_out[n, c, 1, 0] = np.max(input_data[n, c, 2:4, 0:2])
            ref_out[n, c, 1, 1] = np.max(input_data[n, c, 2:4, 2:4])

    print("Max Error:", np.max(np.abs(output_data - ref_out)))