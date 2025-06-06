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
            
        # cache input
        self.padded_input = input
            
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
    
    def backward(self, output_gradient):
        input = self.padded_input # use the cache
        batch_size, in_channels, height, width = input.shape
        _, out_channels, out_height, out_width = output_gradient.shape
        
        padded_gradient = np.zeros((batch_size, self.in_channels, height, width))
        self.kernel_gradient = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.kernel_bias_gradient = np.zeros((self.out_channels,))
        
        for out_x in range(out_height):
            for out_y in range(out_width):
                for kernel_x in range(self.kernel_size):
                    for kernel_y in range(self.kernel_size):
                        # update input gradient
                        padded_gradient[:, :, out_x * self.stride + kernel_x, out_y * self.stride + kernel_y] += np.sum(
                            output_gradient[:, :, out_x, out_y, None] * self.kernel_weight[None, :, :, kernel_x, kernel_y],
                            axis=1
                        )
                        # in this line:
                        # input: batch, in channels
                        # output_gradient: batch, out channels, (new axis)
                        # kernel: (new axis), out channels, in channels
                        
                        # update kernel gradient
                        self.kernel_gradient[:, :, kernel_x, kernel_y] += np.sum(
                            input[:, None, :, out_x * self.stride + kernel_x, out_y * self.stride + kernel_y] * output_gradient[:, :, out_x, out_y][:, :, None],
                            axis=0
                        )
                        # in this line:
                        # kernel gradient: out channels, in channels
                        # input: batch, (new axis), in channels
                        # output_gradient: batch, out channels, (new axis)

        # update kernel bias gradient
        self.kernel_bias_gradient = np.sum(output_gradient, axis=(0, 2, 3))
        return padded_gradient[:, :, self.padding:-self.padding, self.padding:-self.padding] # input gradient

    def gradient_descent(self, learning_rate):
        self.kernel_weight -= learning_rate * self.kernel_gradient
        self.kernel_bias -= learning_rate * self.kernel_bias_gradient
        
        del self.kernel_gradient, self.kernel_bias_gradient

class MaxPooling(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, input):
        # cache input
        self.input = input
        
        batch_size, in_channels, height, width = input.shape
        
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        output = np.empty((batch_size, in_channels, out_height, out_width))
        
        # cache max indices
        self.all_max_indices = np.empty((batch_size, in_channels, out_height, out_width), dtype=int)
        
        for out_x in range(out_height):
            for out_y in range(out_width):
                sliced_input = input[:, :, out_x * self.stride : out_x * self.stride + self.kernel_size, out_y * self.stride : out_y * self.stride + self.kernel_size].reshape(batch_size, in_channels, self.kernel_size * self.kernel_size)
                max_index = np.argmax(sliced_input, axis=2)
                output[:, :, out_x, out_y] = np.max(sliced_input, axis=2)

                self.all_max_indices[:, :, out_x, out_y] = max_index
        
        return output
    
    def backward(self, output_gradient):
        input = self.input
        max_indices = self.all_max_indices
        batch_size, in_channels, height, width = input.shape
        _, _, out_height, out_width = output_gradient.shape
        
        input_gradient = np.zeros_like(input)
        
        for out_x in range(out_height):
            for out_y in range(out_width):
                max_index_mask = np.arange(self.kernel_size * self.kernel_size)[None, None, :] == max_indices[:, :, out_x, out_y][:, :, None]
                max_index_mask = max_index_mask.reshape(batch_size, in_channels, self.kernel_size, self.kernel_size)
                
                input_gradient[:, :, out_x * self.stride : out_x * self.stride + self.kernel_size, out_y * self.stride : out_y * self.stride + self.kernel_size] += output_gradient[:, :, out_x, out_y, None, None] * max_index_mask
            
        return input_gradient
                

class ReLU(Layer):
    def forward(self, input):
        # cache input
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, output_gradient):
        # use cached input
        return np.where(self.input > 0, output_gradient, 0)

class Flatten(Layer):
    def forward(self, input):
        # cache input
        self.input_shape = input.shape
        
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)

    def backward(self, output_gradient):
        # use cached input shape
        return output_gradient.reshape(self.input_shape)

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
        # cache input
        self.input = input
        
        return input @ self.weight + self.bias[None, :]

    def backward(self, output_gradient):
        input = self.input
        # input: (batch_size, in_features)
        # weight: (in_features, out_features)
        # output_gradient: (batch_size, out_features)
        
        self.weight_gradient = input.T @ output_gradient
        self.bias_gradient = np.sum(output_gradient, axis=0)
        return output_gradient @ self.weight.T # input gradient
    
    def gradient_descent(self, learning_rate):
        self.weight -= learning_rate * self.weight_gradient
        self.bias -= learning_rate * self.bias_gradient
        
        del self.weight_gradient, self.bias_gradient

class LogSoftmax(Layer):
    def forward(self, input):
        # numerical trick: subtract the max
        input = input - np.max(input, axis=1, keepdims=True)
        log_softmax = input - np.log(np.sum(np.exp(input), axis=1, keepdims=True))
        
        # cache input
        self.input = input
        self.log_softmax = log_softmax
        return log_softmax
    
    def backward(self, output_gradient):
        softmax = np.exp(self.log_softmax)
        gradient = output_gradient - softmax * np.sum(output_gradient, axis=1, keepdims=True)
        return gradient

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

    def backward(self, output_gradient):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)
        return output_gradient
    
    def gradient_descent(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, 'gradient_descent'):
                layer.gradient_descent(learning_rate)

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
        
        LogSoftmax()
    )
    
def cross_entropy_loss_and_gradient(log_softmax_output, target):
    # log_softmax_output: (batch_size, num_classes)
    # target: (batch_size,)
    
    batch_size, num_classes = log_softmax_output.shape
    mask = np.arange(num_classes)[None, :] == target[:, None]
    
    loss = - np.sum(mask * log_softmax_output) / batch_size
    gradient = - (mask / batch_size)
    return loss, gradient

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
    
    backward_gradient = conv.backward(np.random.randn(*output_data.shape))
    print("Backward gradient shape:", backward_gradient.shape)  # expected: (2, 3, 5, 5)
    print("Weight gradient shape:", conv.kernel_gradient.shape)  # expected: (2, 3, 3, 3)
    print("Bias gradient shape:", conv.kernel_bias_gradient.shape)  # expected: (2,)
    
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
    
    backward_gradient = maxpool.backward(np.random.randn(*output_data.shape))
    print("Backward gradient shape:", backward_gradient.shape)  # expected: (2, 3, 4, 4)
    
    N, C = 2, 3

    ref_out = np.empty((N, C, 2, 2))
    for n in range(N):
        for c in range(C):
            ref_out[n, c, 0, 0] = np.max(input_data[n, c, 0:2, 0:2])
            ref_out[n, c, 0, 1] = np.max(input_data[n, c, 0:2, 2:4])
            ref_out[n, c, 1, 0] = np.max(input_data[n, c, 2:4, 0:2])
            ref_out[n, c, 1, 1] = np.max(input_data[n, c, 2:4, 2:4])

    print("Max Error:", np.max(np.abs(output_data - ref_out)))