# Breakout Game

### Envirnment for task2
The starting state should be the left bottom corner "S" and the terminal state should be the right bottom corner "T". The 3 red cells in Fig. should give a reward of -100 whenever any of them is visited. When the agent reaches the terminal state, a reward of one should be given.

![](./image/maze_environment_task2.PNG)

Q-learning will choose the shortest but more dangerous path "A" and Sarsa will choose a longer but safer path "B" at the end of training.

### Learning Algorithm for task2

Q-learning | SARSA
-----------|---------
Off-policy TD Control | On-policy TD Control

* Off-policy method evaluates or improves a policy different from that used to generate data.
* On-policy method evaluates or improves the policy that is being used to make decisions.

Algorithms are as follows: 
![](https://docs.google.com/drawings/d/e/2PACX-1vSgamZfWbHVk28wnZvfCrBjJuTN8imWkq7mmhAJYaAMV_wIhVDq6n8nfU44bFsdWffYkhvrhqliUHyH/pub?w=1440&h=1080)

hhh


hjgjhgjhgjhgjh

<br>
> This resource is based on EE488C Special Topics in EE <Deep Learning and AlphaGo> Fall 2016, School of EE, KAIST
