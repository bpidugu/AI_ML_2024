# Introduction to  Computer Vision
### Topics
How convolutional neural networks are different from artificial neural networks
Convolutional neural networks architecture
- Convolution Layer
- Pooling Layer
- Fully-connected Layer
---


- The Sobel filter can be implemented using the Sobel() function of OpenCV. It takes in the following parameters:

src - The input image in which we are interested.
ddepth - The depth of the output image. In our code, we use CV_64F, which means a 64-bit floating-point.
dx and dy - The direction in which the edges need to be detected.
ksize - Size of the kernel.

- bitwise_or() function is used to preserve and combine the bright regions from both the filtered outputs whereas the other bitwise functions like bitwise_and would give the bright regions if they are present in both the outputs. Similarly, bitwise_not and bitwise_xor operators cannot be used to combine both the bright regions.

- Convolution under-utilizes the pixels at the end of the image and decreases the dimensions of the image which can be handled with the help of padding and we can also use stride with convolution to record overall information of the features rather than each precise position of features.
- Padding is a simple trick used to add pixels to the edges of an image thus increasing the output shape of an image after convolution.
- Max Pooling is used to find the maximum value of the given patch in an image. Min Pooling is used to find the minimum value; Average Pooling is used to find the average of all the values in a patch and return the average value, whereas Global Average Pooling finds the average over all the values by considering them as a single patch.
- -Padding ensures that the image size is not reduced or the pixels around the image are lost. It tries to maintain the output shape equal to the input shape by using padding=’same’ in the Conv2d function.
- Adding ReLU activation is considered as one of the best practices to add non-linearity to the CNN model.
- The Conv2D, and AveragePooling2D layers are added in the tensorflow.keras.layers module so you can import them as shown below
from tensorflow.keras.layers import Conv2D, AveragePooling2D 
- Convolutional filters require less trainable parameters which gives CNNs a computational advantage.
- The correct flow of the layers in a CNN is: Input Layer -> Convolution Layer -> Pooling -> Flatten-> Fully Connected Layer 

Input Layer: Collects input images

Convolution Layer + Pooling: Build feature maps and extract important features from input images

Flatten: Creates a 1-D array of the output of its previous layer

Fully Connected Layer: Classifies/Predicts the output.
- We do lose information at the pooling layer, but it is only irrelevant informatio
- In the fully-connected neural network, we use flattened outputs from the pooling layer to get the final predictions. Since this is nothing but a dense neural network, operations like weight modification and firing up neurons using activation functions occur. The FC layer does not have the capability of feature extraction.
- 
Encoding the Y values.

In CNN, the image pixel values range from 0-255. Therefore our method of normalization is scaling. Here, we divide all the pixel values by 255 to standardize the images to have values between 0-1.
- model.add(Conv2D()) is creating the first convolutional layer.
64, (3, 3) is the number of filters and kernel size respectively
padding = 'same' provides the output size same as the input size
input_shape denotes input image dimension of the data set. So if the data set has images of dimension 720x720 and have 3 channels, then x = 720 , y = 720 and z = 3

- Pooling Layer has no trainable parameters, so the number of parameters is 0.