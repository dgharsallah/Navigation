# Report

## Learning Algorithm

We used [Deep Q-learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) that uses two techniques: Fixed Q-targets and Experience Replay.
We trained our model for 1000 episodes, 1000 for the number of timesteps per episode and [0.5, 0.01] as a range for epsilon, epsilon-greedy's action selection and a decay of 0.99.


### DQN Hyperparameters
- BUFFER_SIZE = 100000, replay buffer size
- BATCH_SIZE = 64, minibatch size
- GAMMA = 0.99, discount factor
- TAU = 0.001, for soft update of target parameters
- LR = 0.0005, learning rate 
- UPDATE_EVERY = 4, how often to update the network


### Neural Network
The neural network defined in model.py has 3 fully connected layers.
The dimension of the first is state_size * 128, the second is 128 * 128 and the third 128 * action_size.


## Plot of rewards

![Reward Plot](scores.png)

```
Episode 100	Average Score: 1.97
Episode 200	Average Score: 6.35
Episode 300	Average Score: 10.21
Episode 400	Average Score: 11.97
Episode 489	Average Score: 13.03
Environment solved in 489 episodes!	
Average Score: 13.03

```

## Ideas for Future Work
We can improve this algorithm using Prioritized Experience Replay, Dueling Q-Network or Double DQN.



