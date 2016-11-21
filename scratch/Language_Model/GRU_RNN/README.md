This is totally based on <a href="https://github.com/dennybritz/rnn-tutorial-gru-lstm">dennybritz/rnn-tutorial-gru-lstm</a> and <a href="http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/">wildml</a>

Implementation of the Language Model using GRU units in our RNN. Since their implementations are almost identical, it's easy to modify the code to go from GRU to LSTM by changing equations.

<br>
### Adding an embedding Layer
Using word embeddings, the feature can be transformed from low-dimensional to high-dimensional feature vector. It leads to more capture semantic meaning since similar words have similar vectors.

### Adding a second GRU Layer
Adding a second layer to our network allows our model to capture higher-level interactions. However, if you have enough dataset, don't add many GRU layers because it leads overfitting.

