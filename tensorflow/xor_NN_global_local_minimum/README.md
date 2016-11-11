### Confirm global/local minimum from XOR problem

Our problem is to solve XOR problem using simple neural network where there is only 1 hidden layer.
The model diagram is as follows:

![](https://github.com/gritmind/deep_learning_archieves/blob/master/tensorflow/xor_NN_global_local_minimum/image/nn_for_xor.PNG)

The condition for local minimum is as follows:
![](https://github.com/gritmind/deep_learning_archieves/blob/master/tensorflow/xor_NN_global_local_minimum/image/ex_for_local.PNG)

* So, as above image, when a = -1, the model is stuck on local minimum. 
* As you can see code, when we in local minimum, the cost is relatively much higher than in global minimum.

<br>
This resource is based on EE488C Special Topics in EE <Deep Learning and AlphaGo> Fall 2016, School of EE, KAIST
