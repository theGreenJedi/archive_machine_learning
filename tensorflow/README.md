
### Variable type of tensorflow diagram

![](https://docs.google.com/drawings/d/1qaFys5F7_FqI6FvVpQDGIDl-XJwLGh0x3SoL4BTAbw0/pub?w=480&amp;h=360)
* Operator() can be defined as from simple operators such as tf.add(), tf.mul() to complex operators such as tf.reduce_mean(), tf.train.AdagradOptimizer()  
* Operator() consists of Variable(), Constant() and Operator()
* Session.run() can has two kinds of operator(). For example, both tf.reduce_mean() and tf.train.AdagradOptimizer() in RNN  
* Constant() is defined as tf.constant() or any numerical values. 

<br>
### Difference between tf.placeholder and tf.Variable

tf.Variable | tf.placeholder
------------|---------------
initial value | no initial value
contant | feed_dict={tf.placeholder: X_train}
buffer | buffer
dynamic | static (fixed)
weights of a model | train/test dataset

* tf.Variable: have to define an initial value when declaring it
* tf.placeholder: don't have to define an initial value and should be specified at run time with feed_dict argument inside Session.run
* Both tf.Variable and tf.placeholder can be regarded as a buffer because they all has a capsule such as contant, feed_dict={tf.placeholder: X_train} 
* In addition, before using sess.run() for tf.Variable(), we must execute a sess.run(tf.initialize_all_variables()). Just like to notify a setting of variables to session. 

