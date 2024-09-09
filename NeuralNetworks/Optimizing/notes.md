## Topics
- Different Types of Gradient Descent
- Weight Initialization
- Regularization
- Dropout
- Batch Normalization
---
- Neurons are fundamental units of computation in artificial neural networks, processing input data and transmitting signals through weighte connections to produce output.
- **Batch Size** determines the number of samples processed before the model's weights are updated, while **epoch** refers to one complete pass through the entire training datset.
- HeNormal initialization is a weight initialization method by Kaming He, utilizing a Gaussian distribution with a zero mean and variance scaled for Rectified Linear Units (ReLU).
- If the learning rate is too large, Gradient Descent may oscillate or diverge instead of converging to the minimum of the loss function.
- It accelerates the gradient descent algorithm by considering the exponentially weighted average of the gradients.
- In SGD, the weights are updated after computing the gradient for each individual sample, whereas in Gradient Descent, the gradient is computed using the entire dataset before updating the weights.
- An epoch represents one complete pass through the entire training dataset during the training process. It is used to update the model's parameters iteratively to minimize the loss function.
- An iteration typically refers to one update of the model's parameters based on a batch of data. During each iteration, the model computes gradients using the current batch of data, updates its parameters according to the optimization algorithm (e.g., gradient descent), and moves closer to minimizing the loss function. This process repeats until a certain stopping criterion is met, such as reaching a maximum number of iterations or achieving satisfactory performance.
---
## SGD with Momentum & AdaGrad

- Momentum in gradient descent allows for acceleration of the optimization process. By accumulating gradients from previous steps, momentum helps to maintain consistent direction, which can accelerate convergence towards the minimum of the loss function.
- The learning rate in the adagrad decreases for every iteration during the training process, and the learning rate here is dimension-specific.
- In Adagrad, the learning rate is adapted for each parameter individually based on the historical gradients for that parameter. It dynamically decreases the learning rate for parameters that have large gradients, allowing for larger updates for infrequent parameters and smaller updates for frequent parameters. This adaptive learning rate mechanism helps to handle sparse gradients and leads to more stable convergence, as the learning rate effectively decreases as the algorithm progresses.
- Momentum smoothens the optimization trajectory by accumulating past gradients, helping to maintain direction and momentum, which aids in escaping shallow local minima and accelerating convergence along the relevant dimensions.
- AdGrad, an optimization algorithm for neural network training, adjusts the learning rate per paramter based on historical gradients. It addresses the issue of vanishing or exploding gradients, promoting stable and efficient training.
  
  ---
  ## ADAM, RMS Prop

  - Adam combines both momentum and adaptive learning rates. By incorporating moving averages of both gradients and squared gradients, Adam achieves efficient and effective optimization, making it a popular choice in deep learning.
  - RMS Prop - It updates the learning rate using the exponentially decaying average of past squared gradients. RMSProp uses an exponentially-decaying average of past squared gradients to scale the learning rate for each parameter. This helps in adapting the learning rates during training, making it more suitable for non-stationary objectives and improving convergence.
---

## Weight Initialization

- Overfitting coccurs when a ML learning model learn to memorize the training data rather than generalize from it, resulting in poor performance on unseen data.
- Variance mearsures the average degree to which each point differs from the mean of the dataset. It is a statistical metric used to quantify the spread of data points in a datset.

## Batch Normalization

- A batch normalization layer is used in neural networks to normalize the inputs of each layer, reducing internal covariate shift and accelerating the training process.
- It guards against overfitting
- It helps train deep neural networks by stabilizing and accelerating the training process.
- In neural networks, each connection between neurons is associated with a weight. The magnitude of these weights influences the behavior of the network during training and inference.
- Activation refers to the process of transforming the input of a neuron into its output. Activation functions introduce non-linearity to the network, allowing it to learn complex patterns in the data.
- Batch Normalization is a regularization technique applied before activation function or after activation function in a neuron, which normalizes all the input before sending it to the next layer.
- Batch Normalization can’t be applied on the output layer since it will normalize the value, and the output layer will output probabilities and numerical for classification and regression problems respectively.
- Working of Batch Normalization 

Step 1-  Normalization: Normalize X (input) by subtracting its mean and dividing by its standard deviation. So, no learnable parameters in step. 

Step 2 - Scaling: Scale the output of Step 1 by multiplying it with a learnable parameter gamma

 

Step 3 - Shifting: Shift the output of Step 2 by adding an offset (learnable parameter beta) 

So, totally 2 learable parameters

---

## L1/L2 Regularization
- To Guard Against Over-fitting
- Overfitting occurs whena ML model learns to memorize the training data rather than generalize to new, unseen data. It usually happens when a model is too complex relative to the amount of training data.
- The main purpose of L2 regularization in neural networks is to prevent overfitting by penalizing large weights
- L1 regularization works by penalizing the sum of the absolute values of the weights, which promotes sparsity in the learned model. This means that L1 regularization encourages the model to select only the most important features by shrinking the less relevant features' weights towards zero. 
- L2-Regularization is also known as Ridge and L1 regularization as Lasso.
- L1 regularization, also known as Lasso regularization, encourages sparsity by penalizing the absolute value of the coefficients. As a result, it tends to push the coefficients of less important features to exactly zero, effectively performing feature selection by eliminating irrelevant features from the model. On the other hand, L2 regularization, also known as Ridge regularization, penalizes the square of the coefficients but does not inherently lead to feature selection as it tends to shrink the coefficients towards zero rather than setting them exactly to zero.
- ---
## Dropouts
- Correltation measures the strength and direction of the relationship between two variables, It ranges from -1 to 1, with 0 indicating no correlation, 1 indicating a perfect positive correlation, and -1 indicating a perfect negative correlation.
- Dropout is a regularization technique used to prevent the model from overfitting. Dropout drops the number of neurons in the hidden layer according to the dropout ratio given by default ratio is 0.5.
- Dropout technique can not be applied to the output layer since the output layer will be giving the probabilities and numerical values for classification or regression problems respectively, and dropout will randomly drop the neurons in the layer.
- ---
## Neural Network Architectures

- Recurrent Neural Networks (RNNs) are a Neural Network architecture that can be applied to sequential data such as text, audio, video (a sequence of image frames), and time series data. RNNs are named Recurrent Neural Networks because they perform the same recurring task for each element in a sequence, with the outcomes based on prior computations. RNNs can be viewed as having a "memory" that stores information about these prior computations.
- The three main components of an LSTM cell are Input gate, output gate, and forget gate.

An LSTM (Long Short-term Memory) cell has three main components:

The input gate: controls the amount of new information that is allowed to enter the cell state.

The forget gate: controls the amount of information that is allowed to be forgotten from the previous cell state.

The output gate: controls the amount of information that is allowed to flow out of the cell state and into the output.

- The central role of an LSTM model is held by a memory cell known as a ‘cell state’ that maintains its state over time. This memory cell is key to the LSTM retaining a longer-term memory than what was possible with RNNs. This longer-term memory also helps in solving the vanishing gradient problem that happens with RNNs. 
- Convolutional Neural Networks (CNNs) are specifically designed for image recognition tasks and are the most suitable architecture.CNNs excel at capturing spatial hierarchies of features in images through convolutional layers, pooling layers, and fully connected layers. They have been widely successful in tasks such as image classification, object detection, and image segmentation, making them the preferred choice for image recognition problems.
- Autoencoders can indeed be used for dimensionality reduction. By compressing high-dimensional input data into a lower-dimensional latent space, autoencoders effectively perform dimensionality reduction. The encoder part of the autoencoder learns to extract meaningful features from the input data, which are then used to reconstruct the original input in the decoder part. The dimensionality of the latent space represents a compressed representation of the input data, capturing its essential characteristics in a lower-dimensional space.
- Yes, both dropout and batch normalization can be used in the same model architecture. Dropout is a regularization technique that helps prevent overfitting by randomly dropping units during training, while batch normalization normalizes the activations of each layer to ensure stable training. Both techniques can be beneficial for improving the performance and generalization of neural networks, and they can be used together in the same model architecture to achieve better results.
- Different layers require different initialization strategies based on factors such as the activation function, the depth of the network, or the type of data being processed. Therefore, employing different weight initialization techniques within the same model architecture is a common practice in deep learning.
- ---

## FAQ - Optimizing Neural Network
Q1. The various methods for creating Neural Networks models are listed below. What is the difference between these codes? What effect do these codes have on the model?
     model = tf.keras.Sequential()

     model = keras.Sequential() 

     model = Sequential()

The only difference between the three codes mentioned above is in calling the Sequential function, and all three codes will work and produce the same model output.

model = tf.keras.Sequential()
In the above code, the Sequential function is called using the TensorFlow and Keras library.

model = keras.Sequential()

In the above code, the Sequential function is called using the Keras library.

from tensorflow.keras.models import Sequential

model = Sequential()

In the above code, the Sequential function is called using Tensorflow and Keras libraries after importing the libraries.

Q2. Is it always necessary to use batch normalization when building models with ANN? What is its application?
No, Batch Normalization is not always used in the model unless the model is overfitting. Batch Normalization is one of the techniques used in Neural Networks to prevent the model from overfitting. 

Q3. How do we decide how many neurons to include in each of the hidden layers? Why 256, 64, 32, etc.?
There are a few hyperparameters in Neural Networks that we will pass to the model, and the number of neurons in the model is one of them, so there is no rule of thumb for what number to use in the model.

As we know, computers work only with 0's and 1's, which are only two digits. This is why all the memory addressing, resolutions (in games), and size of storage devices are powers of two. And this is the reason for the number of neurons to be powers of two, too. Neural networks require GPUs for faster processing. And there is no rule that we should only use the power of two numbers as the neurons. We can use the non-power of two numbers as well.

Q4. What is the function of 'units' in the code below?
    model2.add(Dense(activation = 'relu', input_dim = 11, units = 128))

 Units in dense layers are the number of neurons present in the dense layer. You can specify the number of neurons directly or define units = 128, which is shown in the below example.

model2.add(Dense(activation = 'relu', input_dim = 11, units = 128))

model2.add(Dense(128, activation = 'relu', input_dim = 11))

Q5. Let’s say there is a dataset with 11 columns and 10000 rows. So the input dim should be equal to the number of columns, and the units should be equal to the number of rows?
Yes, if there are 11 columns in the dataset then the input _dim will be 11, but the units in the input layer will not be equal to the number of rows in the dataset. The number of units that should be present in the input/hidden layer is a hyperparameter that will be passed to the model. The number of units in the hidden layer can be 16,32,64,128,512 and 1024 etc.

Q6. Why is it necessary to apply to_categorical on the target column? Is this due to the Multi-Class classification problem?
Yes, for a multi-class classification problem, we must encode the target column by to_categorical to remove the weightage of the number. 

In the week-1 hands-on, we used a multi-class classification with 10 classes to predict between 0 and 9. When we use this target variable in its current form, the model interprets the highest number, 9, as having more weight than other numbers, and the model is biased toward the highest number. To remove the weightage of the number, we will encode the target variable with to_categorical.

---
- Sigmoid and TanH can be used with Xavier initialization

- ReLU and LeakyReLU can be used with HE initialization
- BatchNormalization is used only after every dense layer and not before the input layer
- BatchNormalization should not be used after the output layer.
- RMSprop is also called Root Mean Square propagation and is an improved version of Adagrad that aims to reduce the aggressive learning rate by taking the exponential average of the gradients instead of the cumulative.
- I. Batch Normalization uses population statistics (mean and variance) computed during the training process to standardize the data during testing. This ensures that the testing process remains consistent with the normalization applied during training and helps improve the stability and performance of the model.

- II. BatchNormalization is typically used within the hidden layers of a neural network, not before the input layer. It's applied to the outputs of hidden layers to normalize the activations and accelerate training. It's not common to use BatchNormalization directly before the input layer.
- The formula for L1 regularization: L(y,yhat) * (the absolute sum of the coefficients)
The formula for L2 regularization: L(y,yhat) * (the square of the magnitude of the coefficients)
- Adam stands for Adaptive moment and is a combination of RMSprop and SGD with momentum, and from RMSprop, it can change/scale the learning rate efficiently, and with the momentum, it uses an exponentially weighted average to avoid noise.
- I. Both L1 and L2 regularization techniques are used to reduce model complexity and prevent overfitting. They add a penalty term to the loss function, discouraging the model from assigning excessively high weights to features.

II. L1 regularization, also known as Lasso regularization, not only assists in reducing overfitting but can also aid in feature selection. It tends to drive some feature weights to exactly zero, effectively excluding those features from the model. This property can be helpful for identifying the most important features.

III. L2 regularization, also known as Ridge regularization, shrinks the coefficients towards zero without making them exactly zero. It helps in reducing model complexity and also mitigates multicollinearity, a situation where predictor variables are highly correlated, which can lead to unstable coefficient estimates.
- If the weights in the neural network are initialized to zero, then the input to the 2nd layer will be the same for all nodes. Then all the neurons will follow the same gradient, and will always end up doing the same thing as one another and ends up learning no new feature.
- Setting the learning rate in Gradient Descent too high can lead to oscillations in the optimization process. The algorithm may start overshooting the minimum, making the parameter updates bounce back and forth around the optimal solution. This oscillation prevents convergence and can cause the algorithm to fail to converge.