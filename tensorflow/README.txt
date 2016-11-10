
## Variable type of tensorflow

### Difference between tf.placeholder and tf.Variable
* tf.Variable: have to define an initial value when declaring it
* tf.placeholder: don't have to define an initial value and should be specified at run time with feed_dict argument inside Session.run

### In terms of Session
* tf.constant: it can be only an argument of Session function
* tf.(operators): 
* tf.Variable
* tf.placeholder

### In terms of 
* tf.constant
* tf.(operators)
* tf.Variable: it's like global variable. we can print it.
* tf.placeholder
