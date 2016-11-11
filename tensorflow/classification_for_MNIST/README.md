
### MNIST dataset
* database of handwritten digits 0~9
* train set: 55000 / validation set: 5000 / test set: 10000
* each image is gray scale / its dimension is 28 pixels by 28 pixels

### Softmax regression
- Softmax regression or multinomial logistic regression is a generalization of logistic regression to the case where we want to handle multiple classes.
- Softmax performs component-wise exponential function with normalization. Therefore the output of the softmax layer behaves as a probability mass function (i.e., each output is between 0 and 1 and the sum of all outputs is 1) and indicates the likelihood of each output.
- Since the model is simple, its performance does not improve much even if we increase the number of epochs. 

### 2-layer CNN + dropout
```
x_image = tf.reshape(x, [-1, 28, 28, 1])
```
First we should reshape the input image x to (batch_size)*28*1. -1 means indefinite, which will be automatically calculated to match the total size, e.g., it becomes equal to the number of examples in a minibatch during training and it becomes equal to the number of examples in the test set during evaluation using the test set. The last dimension indicates the number of channels of an image, i.e., a monochrome image has 1 channel and a full color RGB imgae has 3 channels. 



The accuracy of classification gets better as the neural network becomes more complex.

> This resource is based on EE488C Special Topics in EE <Deep Learning and AlphaGo> Fall 2016, School of EE, KAIST
