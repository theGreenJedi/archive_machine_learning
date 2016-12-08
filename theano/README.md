# Theano

* Symbolic computation leads the program to be concise and fast
* Since symbloic computation can do differentiation, we don't need implement 'Back-propagation' algorithm
* Can use it CPU and GPU in the same code
* It is based on Python

## Symbolic operation
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




