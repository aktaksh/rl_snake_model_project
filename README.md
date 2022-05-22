# Snake game based on Deep Reinforcement Learning with Q Learning , SARSA and EXPECTED SARSA
# Instructions on how to run the Game
## Project: Train AI to play Snake

The goal of this project is to develop an AI bot that can learn  the popular game Snake from scratch and 

We apply a deep reinforcement learning algorithm as a result. This method uses state-related agent parameters as well as positive or negative action-based rewards. There are no rules in the game, and the snake initially has no idea what to do. The agent's purpose is to comprehend the situation and design a strategy to maximize the score or reward. This video demonstrates how a deep reinforcement learning algorithm learns to play snakes. 

After 10 minutes of training, score from 0 to 40 points and demonstrate a sound approach (100 gameplay).We also used Bayesian optimization techniques to discover the best parameters for your deep RL approach and some of the parameters for your deep neural network.


### Comparing Q-Learning, SARSA and Expected SARSA

You can find here implementations of Q-Learning, SARSA and Expected SARSA, used to compare the algorithms. 
The results and final conclusions can be found inside the [results](./results) folder.

## How to Install
This project requires Python 3 with the `pygame` library installed, as well as PyTorch. If you encounter any error with `torch`.

The full list of requirements is in `Requirement_RL.txt`. Latest version of packages have been tested to run smooth.


## How to Run
To run and show the game, execute in the root folder:

Activate  the virtual environment or create your own.

```bash

source ./rl_snake_venv/bin/activate

python3 RLearningSnake.py

```

Arguments/Options available:

- `--display` - default True, display or not the game view
- `--speed` -  default 50, game speed

The default pretrained configuration loads the file `weights/weights.h5` (trained with Expected SARSA on 100 games, best performance among the three algorithms) and runs a test. The data is updated in Project Report

### How to To train the agent, set params in the file `RLearningSnake.py`:

- `params['train'] = True`

The parameters of the Deep Neural Network can be changed in `RLearningSnake.py` from the dictionary `params` in the function `define_parameters()`.

### How to Choose the algorithm:-
 Modify `params['agent_type']` . 
 Values : `'q_learning'`, `'sarsa'` and `'expected_sarsa'`.

Finally, if you run `RLearningSnake.py` from the command line you can set the arguments `--display=False` and `--speed=0`. 
This way, the game display is not shown and the training phase is faster.

### Enable Bayesian Optimizer
Ensure you use correct training with correct optimizer. Incorrect training will lead to failed results.
Modify the parameters accordingly

```bash
python3 RLearningSnake.py --bayesianopt=<boolean>
```

This method uses Bayesian Optimization to optimize some parameters of Deep RL. 
The parameters and the features' search space can be modified as below:
 `SequentialBlackBox.py` by editing the `optim_params` dictionary in `optimize_RL`.


## References

This project code is extended from  (Keras/TensorFlow). https://github.com/python-engineer/snake-ai-pytorch and other relevant work done by many experts in this fields
