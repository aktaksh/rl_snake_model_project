import os
import pygame
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DeepQLearning import DQNAgent
from random import randint
import random
import statistics
import torch.optim as optim
import torch 
from GPyOpt.methods import BayesianOptimization
from SequentialBlackBox import *
import datetime
import distutils.util
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

#################################
#   Define parameters manually  as below #
#################################

# perform one training step!
                # Get batch_size many random samples of state, action, reward, t, next_state from previous
                # experience
                    # Get q_values for both online and offline network
                    # Calculate the td-targets
                    # Perform one update step on the model and 50% chance to switch between online and offline network
                    # Almost the same thing as model.fit in Keras
                    # Update q-target if agent.targetIt is greater than 10
                    # Get q_values for network

                    # Calculate the td-targets td_target = agent.calculate_td_targets(q_values, r, t, gamma) #agent.Q_target
                    # Perform one update step on the model and 50% chance to switch between online and offline network
                    # Almost the same thing as model.fit in Keras
                    # Update q-target if agent.targetIt is greater than 10
def define_parameters():
    params = dict()
    # Neural Network as defined
    params['learning_rate'] = 0.00013526

    params['first_layer_size'] = 200    # neurons in the first layer

    params['second_layer_size'] = 20   # neurons in the second layer

    params['third_layer_size'] = 50    # neurons in the third layer

    params['episodes'] = 100

    params['epsilon_decay_linear'] = 1.0/params['episodes']
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings

    params['weights_path'] = 'weights/weights.h5'
    params['train'] = False
    
    params['test'] = (params['train'] == False)
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'

    params['agent_type'] = 'expected_sarsa' # 'q_learning' | 'sarsa' | 'expected_sarsa'
    return params


class Game:
    """ Initialize PyGAME """
    
    def __init__(self, game_width, game_height, will_display):
        pygame.display.set_caption('Snake as in Reinforcement Learning')

        self.game_width = game_width

        self.game_height = game_height

        self.bg = pygame.image.load("img/background.png")
        if will_display:
            self.gameDisplay = pygame.display.set_mode((game_width + 60 , game_height + 120))
        self.crash = False
        self.player = Player(self)
        self.food = Food()
        self.score = 0


class Player(object):
    def __init__(self, game):

        x = 0.45 * game.game_width

        y = 0.5 * game.game_height

        self.x = x - x % 20
        self.y = y - y % 20

        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False

        self.image = pygame.image.load('img/snake_body.jpg')
        self.x_change = 20
        self.y_change = 0

    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, x, y, game, food, agent):
        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change

        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]

        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]

        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]

        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]

        self.x_change, self.y_change = move_array

        self.x = x + self.x_change
        self.y = y + self.y_change

        if self.x < 20 or self.x > game.game_width - 40 \
                or self.y < 20 \
                or self.y > game.game_height - 40 \
                or [self.x, self.y] in self.position:
            game.crash = True
        eat(self, food, game)

        self.update_position(self.x, self.y)

    def display_player(self, x, y, food, game):

        self.position[-1][0] = x
        self.position[-1][1] = y

        if game.crash == False:
            for i in range(food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))
        else:
            pygame.time.wait(300)


class Food(object):
    def __init__(self):
        self.x_food = 240

        self.y_food = 200

        self.image = pygame.image.load('img/apple.jpg')

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.food_coord(game, player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))


def eat(player, food, game):
    if player.x == food.x_food and player.y == food.y_food:
        food.food_coord(game, player)
        player.eaten = True
        game.score = game.score + 1


def get_record(score, record):
    if score >= record:
        return score
    else:
        return record
def move(action,x,y,d,R,ny,nx):
    #UP = 0
    #RIGHT = 1
    #DOWN = 2
    #LEFT = 3
    end = False
    d = action
    if d == 4:
        if (x-1<0):
            end = True
        else:
            x = x-1
    elif d == 2:
        if (y+1==ny):
            end = True
        else:
            y = y+1
    elif d == 1:
        if (x+1==nx):
            end = True
        else:
            x = x+1
    elif d == 0:
        if (y-1<0):
            end = True
        else:
            y = y-1
    Rew = -10 if (end) else R[x][y][d]
    return x,y,d,Rew
def VI_QL(obs,temp_env):
    nx,ny,nz = obs.shape

    temp_control = temp_env.controller
    temp_grid = temp_control.grid
    snakes_array = temp_control.snakes
    snake = snakes_array[0]

    #Intialize Q-value, Reward and Policy
    Q = np.zeros((nx,ny,4))
    R = np.zeros((nx,ny,4))
    Pol = np.zeros((nx,ny,4))
    Fx,Fy = (0,0)

    #Finding the Food and assigning its Reward to be 1
    for x in range(0,nx,10):
        for y in range(0,ny,10):
            for d in range(4):
                #print(x,y)
                R[x][y][d] = -1
                if (np.array_equal(obs[x][y],FOOD_COLOR)):
                    R[x][y][d] = 10
                    Fx,Fy = x,y

def display_ui(game, score, record):


    myfont = pygame.font.SysFont('sysfont', 20)

    myfont_bold = pygame.font.SysFont('sysfont', 20, True)

    text_score = myfont.render('CURRENT SCORE: ', True, (0, 0, 0))

    text_score_number = myfont.render(str(score), True, (0, 0, 0))

    text_highest = myfont.render('MAX SCORE: ', True, (0, 0, 0))

    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))

    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(player, food, game, record):
    game.gameDisplay.fill((225, 224, 224))
    display_ui(game, game.score, record)
    player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
    food.display_food(food.x_food, food.y_food, game)
    update_screen()


def update_screen():
    pygame.display.update()
    # pygame.event.get() # <--- Add this line (for MAC users) if having problems with UI ###


def initialize_game(player, game, food, agent, batch_size, is_train):
    state_init1 = agent.get_state(game, player, food)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food, agent)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(player, game.crash)
    if is_train:
        agent.remember(state_init1, action, reward1, state_init2, game.crash)
        agent.replay_mem(agent.memory, batch_size)


def plot_seaborn(array_counter, array_score, train):


    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")

    plt.figure(figsize=(13,8))
    fit_reg = False if train== False else True        
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        #color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg = fit_reg,
        line_kws={"color": "#F49F05"}
    )

    # Plot the average line

    y_mean = [np.mean(array_score)]*len(array_counter)
    ax.plot(array_counter,y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='# games', ylabel='score')
    plt.show()


def get_mean_stdev(array):

    return statistics.mean(array), statistics.stdev(array)    

#DDQN = True

#if DDQN is True:

    # Name of weights + Initialize DDQN class + run training + evaluate
#   name_of_weights_DDQN = 'weights_DDQN.h5'
#    pre_agent = DDQNAgent()
#    aft_agent, counter_plot, score_plot = run(pre_agent, name_of_weights_DDQN)
#    aft_eval, counter_plot_eval, score_plot_eval = evaluate_network(aft_agent, name_of_weights_DDQN)
#else:

    # Name of weights + Initialize DQN target class + run training + evaluate
#    name_of_weights_DQN = 'weights_DQN.h5'
#    pre_agent = DQNAgent()
#    aft_agent, counter_plot, score_plot = run(pre_agent, name_of_weights_DQN)
#    aft_eval, counter_plot_eval, score_plot_eval = evaluate_network(aft_agent, name_of_weights_DQN)


def test(params):
    params['load_weights'] = True
    params['train'] = False
    params["test"] = False 
    score, mean, stdev = run(params)
    return score, mean, stdev

# While the snake as not died
            # Get the q-values w.r.t. the states
            # Evaluate new policy w.r.t q-values and epsilon (epsilon-greedy policy)
            # Choose an action w.r.t. the policy and converts it to one-hot format (eg 2 to [0, 0, 1]) action_hot = to_categorical(action, num_classes=3) # [0] # CHANGE HERE
            # Let the player do the chosen action
            # Update the state variables
            # Give reward, 0 - alive, 10 - eaten, -10 - died
            # Store data to the Tuple subclass Transition and then use the function add in replay_buffer to add
            # Transition to replay_buffer.__buffer
            # Assign the "old" state as the new
            # Update record (high score)
            # If we have done more than 1000 steps in total (over multiple games)
            # then !perform one training step!
def run(params):
    """
    Run the DQN algorithm, based on the parameters previously set.   
    """
    pygame.init()
    agent = DQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    total_score = 0
    while counter_games < params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Initialize classes
        game = Game(440, 440, params['display'])
        player1 = game.player
        food1 = game.food

        # Perform first move
        initialize_game(player1, game, food1, agent, params['batch_size'], params['train'])
        if params['display']:
            display(player1, food1, game, record)
        
        steps = 0       # steps since the last positive reward
        while (not game.crash) and (steps < 100):
            if not params['train']:
                agent.epsilon = 0.01
            else:
                # agent.epsilon is set to give randomness to actions
                agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

            # get old state
            state_old = agent.get_state(game, player1, food1)

            # perform random actions based on agent.epsilon, or choose the best action (epsilon-greedy strategy)
            final_move = agent.get_epsilon_greedy_action(state_old)

            # perform new move and get new state
            player1.do_move(final_move, player1.x, player1.y, game, food1, agent)
            state_new = agent.get_state(game, player1, food1)

            # set reward for the new state
            reward = agent.set_reward(player1, game.crash)
            
            # if food is eaten, steps is set to 0
            if reward > 0:
                steps = 0
                
            if params['train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, game.crash)

            record = get_record(game.score, record)
            if params['display']:
                display(player1, food1, game, record)
                pygame.time.wait(params['speed'])
            steps+=1
        if params['train']:
            agent.replay_mem(agent.memory, params['batch_size'])
        counter_games += 1
        total_score += game.score
        print(f'Game {counter_games}      Score: {game.score}')
        score_plot.append(game.score)
        counter_plot.append(counter_games)
    mean, stdev = get_mean_stdev(score_plot)
    if params['train']:
        model_weights = agent.state_dict()
        torch.save(model_weights, params["weights_path"])
    if params['plot_score']:
        plot_seaborn(counter_plot, score_plot, params['train'])
    return total_score, mean, stdev

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()

    parser = argparse.ArgumentParser()

    params = define_parameters()

    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)

    parser.add_argument("--speed", nargs='?', type=int, default=50)

    parser.add_argument("--bayesianopt", nargs='?', type=distutils.util.strtobool, default=False)

    args = parser.parse_args()

    print("Args", args)
    params['display'] = args.display
    params['speed'] = args.speed

    if args.bayesianopt:
        SequentialBlackBox = BayesianOptimizer(params)
        SequentialBlackBox.optimize_RL()

    if params['train']:

        print("Training...")
        params['load_weights'] = False   # when training, the network is not pre-trained
        total_score, mean, stdev = run(params)
        print("total_score:", total_score, " | mean:", mean, "| stdev:", stdev)

    if params['test']:
        
        print("Testing...")
        params['train'] = False
        params['load_weights'] = True
        total_score, mean, stdev = run(params)
        print("total_score:", total_score, " | mean:", mean, "| stdev:", stdev)