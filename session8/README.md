# CIFAR-10 Image Classification with Data Augmentation and Depthwise Convolutions

This repository contains code for training a convolutional neural network (CNN) model on the CIFAR-10 dataset. The model utilizes advanced techniques such as data augmentation, depthwise separable convolutions, and adaptive learning rate schedules to improve its accuracy and generalization. Below is an explanation of the key components of the code, the process that takes place during training, and the technical concepts used.

## Setup and Requirements

To begin, make sure to install the necessary libraries by running the following:

```
pip install torch torchvision tqdm albumentations
```

## Data Preprocessing and Augmentation

### CIFAR-10 Dataset
The *CIFAR-10 dataset* is a collection of 60,000 images in 10 classes, with each image being 32x32 pixels. The dataset is split into 50,000 training images and 10,000 test images.

### Augmentation
Data augmentation is used to artificially increase the size of the training dataset by applying transformations to the original images. This helps in improving model generalization and reducing overfitting. The following augmentations are applied using the Albumentations library:

**Horizontal Flip**: Randomly flips images horizontally with a probability of 50%.
**Shift, Scale, and Rotate**: Randomly shifts, scales, or rotates the image to introduce variations.
**Coarse Dropout**: Randomly drops patches of the image to simulate occlusions and force the model to focus on other parts of the image.
**Center Crop**: Crops the center of the image to a fixed size.
**Normalization**: Normalizes the pixel values based on the CIFAR-10 dataset statistics (mean and standard deviation).
**ToTensorV2**: Converts images to PyTorch tensors, making them compatible with the model.

``` python
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.2),
    A.CoarseDropout(p=0.2, max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465)),
    A.CenterCrop(height=32, width=32, always_apply=True),
    A.Normalize((0.4914, 0.4822, 0.4465), (0.2463, 0.2428, 0.2607)),
    ToTensorV2(),
])
```



## DataLoader
DataLoaders are used to load the data in batches for efficient training. They shuffle the data and ensure parallel data loading using multiple worker threads. The following code snippet creates the data loaders for both the training and testing datasets.
``` python
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
```

## Model Architecture

**Convolutional Neural Network (CNN)**
The model architecture consists of multiple convolutional blocks followed by a fully connected output layer. Each convolutional block contains:

*Convolutional Layers*: For extracting features.
*Batch Normalization*: To normalize activations and improve convergence.
*ReLU Activation*: To introduce non-linearity.
*Dropout*: To reduce overfitting.
*Depthwise Separable Convolutions*: Introduced later for more efficient feature extraction.
Depthwise Separable Convolutions
Depthwise separable convolutions are an efficient alternative to regular convolutions. They split the convolution into two parts:

*Depthwise Convolution*: Applies a single filter per input channel.
*Pointwise Convolution*: Applies a 1x1 convolution to combine the outputs of the depthwise convolution.
This reduces the computational complexity significantly and makes the model more lightweight.
``` python
def depthwise_separable_conv(self, in_channels, out_channels, kernel_size=3, padding=1):
    depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False)
    pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=1, padding=0, bias=False)
    return nn.Sequential(depthwise_conv, pointwise_conv)
```
#### Receptive Field
The receptive field refers to the area of the input image that affects a particular activation in the output feature map. In this model, the receptive field increases as we move through deeper layers because of the combination of kernel sizes and strides used in the convolution operations.

### Final Output
The model ends with an adaptive max pooling layer to compress the feature map and a 1x1 convolution that reduces the output channels to 10, corresponding to the 10 classes in CIFAR-10. The output is then passed through a log_softmax activation to output class probabilities.

``` python
self.out = nn.Sequential(
    nn.AdaptiveMaxPool2d(1),
    nn.Conv2d(64, 10, kernel_size=1),
)
```
**Training Process**

*FitEvaluate Class*
The FitEvaluate class handles the training and evaluation process. It defines methods for:

- Training: The train method performs forward and backward passes, calculates loss, and updates the model parameters.
- Testing: The test method evaluates the model on the test dataset and computes the accuracy.
- Epoch Training: The epoch_training method iterates over epochs, calling the train and test methods after each epoch.


**Optimizer and Scheduler**
The model is trained using the Stochastic Gradient Descent (SGD) optimizer with momentum. A StepLR scheduler is used to decrease the learning rate by a factor of 0.6 every 5 epochs.
``` python
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
scheduler = StepLR(optimizer, step_size=5, gamma=0.6)
```