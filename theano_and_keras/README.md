# Theano
* made by [LISA Lab](http://deeplearning.net/software/theano/)
* It is based on Python, which enable to interwork with other useful python packages such as numpy, scipy, matplotlib, ipython and so on
* Symbolic computation leads the program to be concise and fast
* Since symbloic computation can do differentiation, we don't need implement 'Back-propagation' algorithm
* Can use it CPU and GPU in the same code

## Symbolic Variable
#### 1. Example of Symbolic operation
```ex) y = 5x^2 + 20```
<br>
<br>
Python
```
  def compute(x)
     y = 5*x^2 + 20
     return y
  compute(7)
```
Theano
```
  x = T.scalar()   # Definition of symbolic variable
  y = 5*x^2 + 20   # Symbolic Expression
  compute = theano.function([x], y)   # Compile
  compute(7)
```
#### 2. Example of Symbolic differential operation
```ex) y = 5x^2 + 20```
<br>
<br>
Python: we have to manually differentiate by ourselves
```
  def diff(x):
     y = 4*x + 5
     return y
  diff(7)
```
Theano: a derivative is automatically calculated by Symbolic differential operation
```
  x = T.scalar()
  y = 5*x^2 + 20
  y_prime = T.grad(y, x) # Symbolic differential operation
  diff = theano.function([x], y_prime)
  diff(7)
```
We easily implement complex 'back-propagation' by using Symbolic differential operation

## Shared Variable
#### 1. Shared: to move data from CPU-RAM to GPU-VRAM
```
shared_var = theano.shared(numpy_array)
numpy_array = shared_var.get_value()
```
Data defined as a shared variable can come in and out CPU_RAM and GPU_VRAM
So, when we want to process data in GPU mode, we define the data as the shared variable
#### 2. Givens: to apply shared data to symbolic variable 
:```ex) y = 2*x and we want to apply x to 7``` 
```
# Method 1
  compute = theano.function([x], 2*x);
  compute(7) # When executing this line -> RAM -> VRAM -> GPU Operation
# Method 2
  x_value = theano.shared(7)
  compute = theano.function([], 2*x, givens=[x, x_value]) # When executing this line -> VRAM -> GPU Operation
  compute()
```
#### 3. Updates: to modify shared data by using results of GPU Operation
```
x_val = theano.shared(0)
increase = theano.function([], x_val, updates=(x_val, x_val+1))
increase() # When executing this line -> x_val increases by 1 in GPU-VRAM (Don't need RAM)
```
<br>

# Keras
* It is deep learning framework based on Theano. [here](http://keras.io)
* Offer various deep learning models and optimizers
* We can construct deep learning models just like LEGO Block (similar to Caffe, Torch)
* It is easy to add new models because of theano
* we can save trained model by HDF5 or Json format.

### Example of Multi-layer Perceptron 
The model explanation:
   * 20 nodes in input layer
   * 64 nodes in first hidden layer
   * 64 nodes in second hidden layer
   * 2 nodes in output layer
   * use dropout and stochastic gradient descent optimizer
```
  model = Sequential()
  model.add(Dense(64, input_dim=20, init='uniform', activation='tanh'))
  model.add(Dropout(0.5))
  model.add(Dense(64, init='uniform', activation='tanh'))
  model.add(Dropout(0.5))
  model.add(Dense(2, init='uniform', activation='softmax'))

  sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='mean_squared_error', optimizer=sgd)
```




## Acknowledgement 
> [Biointelligence Laboratory](http://bi.snu.ac.kr) & Department of Computer Science and Engineering & Seoul National University & Eun-Sol Kim
