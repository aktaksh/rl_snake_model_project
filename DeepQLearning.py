import random
import numpy as np
import pandas as pd
from operator import add
import collections
from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'


# define one deep Q-networks
        # define ops for updating the networks
        # Count numbers of iterations without a update of Q-target
        # fc = Dense(120, activation='relu', kernel_initializer='VarianceScaling')(fc)
        # fc = Dense(60, activation='relu', kernel_initializer='VarianceScaling')(fc)
        # trace of taken actions
        # Define second offline network
class DQNAgent(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']        
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.optimizer = None
        self.network()
        self.agent_type = params['agent_type']
        if params['train']:
            supported_agent_types = {
                'q_learning': 'Q-Learning',
                'sarsa': 'SARSA',
                'expected_sarsa': 'Expected SARSA',
            }
            if self.agent_type in supported_agent_types:
                print('Using', supported_agent_types[self.agent_type])
            else:
                print('Agent "', self.agent_type, '" not found, using default Q-Learning instead', sep='')
                self.agent_type = 'q_learning'
          
    def network(self):
        # Layers
        self.f1 = nn.Linear(11, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 3)
        # weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x
    
    def get_state(self, game, player, food):
        """
        Return the state.
        The state is a numpy array of 11 values, representing:
            - Danger 1 OR 2 steps ahead
            - Danger 1 OR 2 steps on the right
            - Danger 1 OR 2 steps on the left
            - Snake is moving left
            - Snake is moving right
            - Snake is moving up
            - Snake is moving down
            - The food is on the left
            - The food is on the right
            - The food is on the upper side
            - The food is on the lower side      
        """
        state = [
            (player.x_change == 20 and player.y_change == 0 and ((list(map(add, player.position[-1], [20, 0])) in player.position) or
            player.position[-1][0] + 20 >= (game.game_width - 20))) or (player.x_change == -20 and player.y_change == 0 and ((list(map(add, player.position[-1], [-20, 0])) in player.position) or
            player.position[-1][0] - 20 < 20)) or (player.x_change == 0 and player.y_change == -20 and ((list(map(add, player.position[-1], [0, -20])) in player.position) or
            player.position[-1][-1] - 20 < 20)) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add, player.position[-1], [0, 20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.game_height-20))),  # danger straight

            (player.x_change == 0 and player.y_change == -20 and ((list(map(add,player.position[-1],[20, 0])) in player.position) or
            player.position[ -1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],
            [-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == -20 and player.y_change == 0 and ((list(map(
            add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,20])) in player.position) or player.position[-1][
             -1] + 20 >= (game.game_height-20))),  # danger right

             (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],[20,0])) in player.position) or
             player.position[-1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == -20 and ((list(map(
             add, player.position[-1],[-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (
            player.x_change == -20 and player.y_change == 0 and ((list(map(add,player.position[-1],[0,20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.game_height-20))), #danger left


            player.x_change == -20,  # move left
            player.x_change == 20,  # move right
            player.y_change == -20,  # move up
            player.y_change == 20,  # move down
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y  # food down
        ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)
# Setting rewards
    def set_reward(self, player, crash):
        """
        Return the reward.
        The reward is:
            -10 when Snake crashes. 
            +10 when Snake eats food
            0 otherwise
        """
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        """
        Store the <state, action, reward, next_state, is_done> tuple in a 
        memory buffer for replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay_mem(self, memory, batch_size):
        """
        Replay memory.
        """
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            self.train_short_memory(state, action, reward, next_state, done)

    def get_epsilon_greedy_action(self, state_old):
        """
        Return the epsilon-greedy action for state_old.
        """
        if random.uniform(0, 1) < self.epsilon:
            # return a random action
            final_move = np.eye(3)[randint(0,2)]
        else:
            # choose the best action based on the old state
            with torch.no_grad():
                state_old_tensor = torch.tensor(state_old.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
                prediction = self(state_old_tensor)
                final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]
        return final_move
# Get target 
    def get_target(self, reward, next_state):
        """
        Return the appropriate TD target depending on the type of the
        agent (Q-Learning, SARSA or Expected-SARSA).
        """
        next_state_tensor = torch.tensor(next_state.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
        q_values_next_state = self.forward(next_state_tensor[0])
        if self.agent_type == 'q_learning':
            target = reward + self.gamma * torch.max(q_values_next_state) # Q-Learning is off-policy
        elif self.agent_type == 'sarsa':
            next_action = self.get_epsilon_greedy_action(next_state) # SARSA is on-policy
            q_value_next_state_action = q_values_next_state[np.argmax(next_action)]
            target = reward + self.gamma * q_value_next_state_action
        elif self.agent_type == 'expected_sarsa':
            probabilities_for_actions = np.array([self.epsilon/3, self.epsilon/3, self.epsilon/3])
            q_values_next_state_numpy = q_values_next_state.detach().cpu().numpy()
            best_action_index = np.argmax(q_values_next_state_numpy)
            probabilities_for_actions[best_action_index] += 1 - self.epsilon
            expected_next_q_value = np.dot(probabilities_for_actions, q_values_next_state_numpy)
            target = reward + self.gamma * expected_next_q_value
        else:
            raise ValueError('agent_type in get_target should necessarily be one of the supported agent types')
        return target

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        state_tensor = torch.tensor(state.reshape((1, 11)), dtype=torch.float32, requires_grad=True).to(DEVICE)
        if not done:
            target = self.get_target(reward, next_state)
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()