# Breakout Game
We train an agent to play a simple version of Breakout. There are 25 bricks in 5 rows and 5 columns. At each time, the agent can take one of 5 actions to remove the bottom brick in one of 5 columns. Unlike the original Breakout, there is no ball and thus you don't need to worry about losing a ball. A reward of 1 is given for removing a brick.

## breakout1
![](./image/environment1.png)
Figure. Example of state transition in environment 1. <br>
In environment 1, if any one of 5 columns is cleared, then all the remaining bricks are removed and you get a reward equal to the number of removed bricks. Therefore, an optimal policy for this environment is to clear any one of 5 columns. 

## breakout2
![](./image/environment2.png)
Figure. Example of state transition in environment 2. <br>
In environment 2, if any one of 5 columns is cleared, then all the remaining bricks in the upper 4 rows are removed and you get a reward equal to the number of removed bricks. Therefore, an optimal policy for this environment is to clear any one of 5 columns and then remove the remaining 4 bricks at the bottom row.

## breakout3
![](./image/environment3.png)
Figure. Example of state transition in environment 3. <br>
In environment 3, if any one of 5 columns is cleared, then all the remaining bricks are removed and you get a reward equal to the number of removed bricks. Therefore, an optimal policy for this environment is to clear any two consecutive columns. 

### About all breakout1,2,3 ...
* For all 3 environments, the total accumulate rewards is always 25 at the end of a game. Therefore, the goal is not to get a higher score, but to finish the game as soon as possible. If the discount factor 𝛾 is strictly less than 1, then reinforcement learning will try to learn to finish the game as quickly as possible.
* Since there are 6 possible states for each column, i.e., empty, only the top brick is present, only the top two bricks are present, …, and all 5 bricks are present, the total number of states is 6^5.
* A state is represented 3 different ways.
   * For storing a state in replay memory in DQN, we use a vector representation, i.e., s=[s_1,s_2,s_3,s_4,s_5], where s_i is 0~5 indicating the number of remaining bricks in the i-thcolumn.
   * For feeding a state to the neural network in DQN, we use a matrix representation that is 5x5 matrix of 0’s and 1’s indicating the presence of each brick. A vector representation can be converted to an equivalent matrix representation by using the function “matrix_state” in “state_representation.py”.
   * For Q learning, we use a scalar representation, i.e., s=0,…,6^5−1, where s=0 means no bricks and s=6^5−1means all bricks are present. A vector representation can be converted to an equivalent scalar representation by using the function “scalar_state” in “project2_state_representation.py”.
   
### DQN breakout1
The neural network has only one layer, but it can learn to play optimally with only **10** episodes. It is possible for such a simple neural network to produce an optimal policy.

### DQN breakout2
The neural network still has only one layer. It takes about **5,000 ~ 10,000** episodes to train the network. More episodes are needed than breakout1 since this game is more complicated.

### DQN breakout3
The neural network still has only one layer. It now takes about **20,000** episodes to train the network.

<br>
## Code details of Deep Q network 
Basic algorithm is the same as in [[3]](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html). We use an online network and a target network. Initially, they are both identical copies of each other. Then, by using the experiences of the agent, the online network is updated. Then, after some iterations, the target network is updated by copying the parameters of the online network to the target network. In the DQN, we can use experience replay as done in [[3]](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html). We store the state, action, new state and reward in the memory and sample them in batch when we need parameter update. Replay memory size is set to be big enough to store the whole experience during training.

### 1. Q network architecture (def DeepQNetwork)
Q network consists of single fully connected layer, where we get the image of the state as a matrix and output the Q values for all possible actions.
```
weights1 = tf.get_variable("weights1", [5 * 5, 5], initializer = tf.random_normal_initializer())
biases1 = tf.get_variable("biases1", [5], initializer = tf.random_normal_initializer())
```
These are the weights and biases for the network
```
in1 = tf.reshape(state, [-1, 5*5])   # It is 'input Layer'
```
This reshapes the state to a length-25 vector. -1 means undecided, which will be automatically set later to be equal to the batch size when you feed data to the neural network.
```
return tf.matmul(in1, weights1) + biases1
```
This defines the output Q values without activation function.
```
QN, state_dim= DeepQNetwork, [None, 5, 5]
state_representation= matrix_state
```
We set the Q network and the input state dimension as 5 x 5. ‘state_representation’ function converts the state vector to a matrix form as explained before.

### 2. Placeholder and optimizer design
We use five placeholders S,A,R,Sn, and T for current state, action, reward, next state, and indicator for episode termination, respectively. The shape of each placeholder is set properly. [None] means unspecified, which will be set later to match the batch size.
```
online_Q= QN(S)
target_Q= QN(Sn)
```
This specifies ‘online_Q’ is the output of the online network with input ‘S’.
This specifies ‘target_Q’ is the output of the target network with input ‘Sn’.
```
online_net_variables= tf.get_collection(tf.GraphKeys.VARIABLES, scope = "online_net")
target_net_variables= tf.get_collection(tf.GraphKeys.VARIABLES, scope = "target_net")
```
We gather the variables such as weights and biases that contribute to the update of online_Qand target_Q.
```
Y_targets= R + gamma * tf.mul(T, tf.reduce_max(target_Q, 1))
```
This corresponds to the R+max(Q(sn,:)) in the Q-learning. ‘T’ is multiplied since we only need to have ‘R’ if ‘Sn’ is a terminal state.
```
Y_onlines= tf.reduce_sum(tf.mul(online_Q, A), 1)
```
This extracts the Q value for the current state and action.
```
loss = tf.reduce_mean(tf.square(tf.sub(Y_targets, Y_onlines)))
```
This is the loss function defined as E[(Q(s,a) - (R+gamma x max(Q(s',:)))^2]
```
optimizer = tf.train.RMSPropOptimizer(alpha).minimize(loss, var_list= online_net_variables).
```
Finally, we train the weights and biases of the neural network to minimize the loss. RMSPropOptimizeris a well-known optimize, which is also used in DQN in [[3]](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html). This optimization is performed by ‘online_net_update’ function.
```
target_net_update= [target_net_variables[i].assign(online_net_variables[i]) for i in range(len(online_net_variables))]
```
This updates the target network by copying the variables of the online network.

### 3. Main training algorithm
Training is done similarly as in [[3]](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html). Here, we explain things that are specific to our implementation. ‘replay_memory’ and ‘minibatch’ are initialized to be lists with 5 empty lists. online_net_variablesand target_net_variablesare initialized.
```
epsilon = 1 -float(i_episode) / (num_episodes–1)
```
We linearly decrease epsilon from 1 in the first episode to 0 in the final episode.
<br>
For each episode, we initialize the state ‘S(underbar)’, indicator of the terminal state ‘T(underbar)’, and time stamp ‘time_episode_start’.
```
Q_ = sess.run(online_Q, feed_dict= {S: state_representation([S_])})
```
Obtain Q values for the current state 'S(underbar)' using the online network. 
<br>
After we take action and observe the new state, reward, and indicator for a terminal state, we store the observed values into the replay memory. Then, we ‘num_sampling’ random samples from the replay_memoryand train the Q network. If ‘replay’ variable is set to 0, only the most recent observed sample is used for training. if target_net_update_counter< target_net_update_period… sess.run(target_net_update)
This is for updating the target network.


## Acknowledgment
> EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016 & Information Theory & Machine Learning Lab, School of EE, KAIST & Wonseok Jeon and Sungik Choi (wonsjeon@kaist.ac.kr, si_choi@kaist.ac.kr)
