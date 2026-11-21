# Simple CNN

A convolutional neural network implemented from scratch in modern C++ with no external ML libraries. It trains on CIFAR-10 and classifies 32x32 RGB images into 10 classes.

## Architecture

```
32x32 RGB image
  -> 8 filters, 3x3, stride 1, ReLU        -> 8 feature maps, 30x30
  -> 2x2 max pooling, stride 2             -> 8 pooled maps, 15x15
  -> flatten                                -> 1,800 inputs
  -> MLP 1800 -> 80 -> 10, softmax          -> 10 class probabilities
```

- One convolutional layer, then one max-pooling layer, then a predictor MLP.
- Filters are initialized with He initialization: `N(0, sqrt(2 / fan_in))`, where `fan_in = channels * filterSize^2`.
- Loss is cross-entropy, via `softmax` output and `predicted - target` gradients.
- Training is batch-1 SGD with a fixed learning rate of `0.0005`.

## How backprop works here

1. The predictor MLP backpropagates through softmax + cross-entropy, then through its ReLU hidden layer, giving gradients with respect to the flattened pooled vector.
2. Max pooling routes each gradient only to the position that was the maximum in its 2x2 region. The other three positions get zero. The argmax positions are saved during the forward pass.
3. ReLU masks the gradient: it passes through only where the pre-activation was positive.
4. The filter gradient is the input patch dotted with the upstream gradient, summed over every position the filter visited. The bias gradient is the sum of the upstream gradients for that filter.

## Files

- `main.cpp` — CIFAR-10 loading, training loop, held-out validation, and a live terminal prediction display.
- `convolutional_neural_network.hpp` / `.cpp` — convolution, ReLU, max pooling, flattening, and filter/bias gradients.
- `neural_network.hpp` / `.cpp` — the predictor MLP: forward, backprop, softmax, He initialization, SGD updates.

## Build

Requires a C++26-capable compiler (for `std::mdspan` and other modern standard-library features), e.g. GCC 14+.

```bash
g++ -std=c++26 -O2 -o CNN main.cpp convolutional_neural_network.cpp neural_network.cpp
```

No CMake file is included yet. No external libraries are needed.

## Dataset

Download the CIFAR-10 binary version and extract it so the folder `cifar-10-batches-bin` sits next to the executable (or pass its path as the first argument).

```bash
./CNN [path-to-cifar-10-batches-bin] [frame_delay_ms=2300]
```

## Training

Training runs indefinitely until you press ENTER. Every 150 images the screen refreshes with images seen, epoch, validation accuracy, and cross-entropy loss.

Validation is a fixed set of 128 training images that are never passed to `fit()`, evaluated every 750 images. After you press ENTER, the program cycles through unseen CIFAR-10 test images and shows each prediction with its confidence. Press ENTER again to exit.

## Limitations

- Single convolutional layer, no padding, no second conv layer.
- Batch-1 SGD only: no minibatching, no Adam, no learning-rate schedule, no weight decay.
- Inputs are only scaled to `[0, 1]`; there is no per-channel mean/std normalization and no data augmentation.
- No final test-set accuracy is printed — the test phase is a per-image prediction slideshow.
- The live display is Linux-terminal only (`poll` / `termios`).
