# Jun 4: Implementation of forward for Convolutional and Pooling layers

## Convolutional Layer

The input tensor has shape: `(batch_size, in_channels, height, width)`. Each image in the batch may have multiple input channels (e.g., RGB has 3), and convolution is performed per-channel and then summed across.

![](./conv-demo.png)

### Step 1: Padding

We use zero padding for the padding operation. The effective shape of the input tensor after padding will be `(batch_size, in_channels, height + 2 * padding, width + 2 * padding)`.

```python
if self.padding > 0:
    padded_input = np.zeros((batch_size, in_channels, height + 2 * self.padding, width + 2 * self.padding))
    padded_input[:, :, self.padding:-self.padding, self.padding:-self.padding] = input
    input = padded_input
```

### Step 2: Output size

```python
out_height = (height - self.kernel_size) // self.stride + 1
out_width = (width - self.kernel_size) // self.stride + 1
output = np.zeros((batch_size, self.out_channels, out_height, out_width))
```

This utilizes the formula for calculating the output size of a convolutional layer:
$$
\text{Output Size} = \left\lfloor \frac{\text{Input Size} + 2 \cdot \text{Padding} - \text{Kernel Size}}{\text{Stride}} \right\rfloor + 1
$$

Output is created as a zero tensor.

### Step 3: Sliding Window Convolution

Reference image: Vincent Dumoulin’s convolution arithmetic visualizations

![](https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/same_padding_no_strides.gif)

```python
for out_x in range(out_height):
    for out_y in range(out_width):
        for kernel_x in range(self.kernel_size):
            for kernel_y in range(self.kernel_size):
                output[:, :, out_x, out_y] += np.sum(
                    input[:, None, :, out_x * self.stride + kernel_x, out_y * self.stride + kernel_y]
                    * self.kernel_weight[None, :, :, kernel_x, kernel_y],
                    axis=2
                )
        output[:, :, out_x, out_y] += self.kernel_bias[None, :]
```

The key idea: for each `(out_x, out_y)` output position, extract a sliding window from the input and multiply with the filter weights, summing over input channels. The value at `(out_x, out_y)` for the output is contributed by the kernel value at `(kernel_x, kernel_y)` and the input value at `(out_x * stride + kernel_x, out_y * stride + kernel_y)`.

Here, broadcasting is used to accelerate the multiplication. The shape is specified below:

- `input`: the element at the position has shape `(batch_size, in_channels)`. After adding a new axis, it becomes `(batch_size, 1, in_channels)`.
- `kernel_weight`: the element at the position has shape `(out_channels, in_channels)`. After adding a new axis, it becomes `(1, out_channels, in_channels)`
- `output`: the element has shape `(batch_size, out_channels)`

Summing over `axis=2` is done to collapse the input channels, resulting in the output shape of `(batch_size, out_channels)`.

## Pooling Layer

## Other Layers