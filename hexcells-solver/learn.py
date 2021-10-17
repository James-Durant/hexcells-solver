import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

from solve import Solver

from grid import Cell
import datetime

from tensorflow.keras.callbacks import TensorBoard

from navigate import Navigator
from parse import GameParser
from window import get_window

class Environment:
    STATE_DIMS = (7, 3)
    MAX_HEXES = 10
    
    def __init__(self, parser):
        self.__parser = parser
        self.__solver =  Solver(parser)
        self.__grid = self.__parser.parse_grid()
        self.__state = self.__get_state()
    
    def get_state(self):
        return self.__state
    
    def __get_state(self):
        encoding = np.zeros(Environment.STATE_DIMS)
        
        for row in range(self.__grid.rows):
            for col in range(self.__grid.cols):
                cell = self.__grid[row, col]
                if not cell:
                    rep = -1
                elif cell.colour == Cell.BLACK:
                    if cell.digit == '?':
                        rep = 7
                    else:
                        rep = cell.digit
                
                elif cell.colour == Cell.ORANGE:
                    rep = 8
                    
                elif cell.colour == Cell.BLUE:
                    rep = 9
      
                encoding[row, col] = rep
            
        #print(encoding)
        return encoding #(encoding, self.__grid.remaining)
        
    def step(self, action_index):
        button = action_index // Environment.MAX_HEXES
        action_index %= Environment.MAX_HEXES
        
        row = action_index // Environment.STATE_DIMS[1]
        col = action_index % Environment.STATE_DIMS[1]
        
        reward = -1
        solved = False
        if row in range(self.__grid.rows) and col in range(self.__grid.cols):
            cell = self.__grid[row, col]
            if cell and cell.colour == Cell.ORANGE:
                valid_left_click_cells, valid_right_click_cells = self.__solver.solve_single_step(self.__grid)
                
                if ((button == 0 and cell in valid_left_click_cells) or
                    (button == 1 and cell in valid_right_click_cells)):
                    
                    left_click_cells = [cell] if button == 0 else []
                    right_click_cells = [cell] if button == 1 else []

                    solved = len(self.__grid.unknown_cells()) == 1
                    if solved:
                        self.__parser.click_cells(left_click_cells, 'left')
                        self.__parser.click_cells(right_click_cells, 'right')
                        remaining = 0
                    else:
                        remaining = self.__parser.parse_clicked(self.__grid, left_click_cells, right_click_cells)
                   
                    if remaining is not None:
                        self.__grid.remaining = remaining #self.__state[1]
                    
                    if cell.colour == Cell.BLACK:
                        if cell.digit == '?':
                            rep = 7
                        else:
                            rep = cell.digit
                    
                    elif cell.colour == Cell.ORANGE:
                        rep = 8
                        
                    elif cell.colour == Cell.BLUE:
                        rep = 9
                    
                    self.__state[row, col] = rep
                    reward = 1
        
        #print(reward)
        return self.__state, reward, solved
    
    def unknown(self):
        action_indices = []
        for cell in self.__grid.unknown_cells():
            row, col = cell.grid_coords
            index = Environment.STATE_DIMS[1]*row+ + col
            action_indices.append(index)
            action_indices.append(index+Environment.MAX_HEXES)
            
        return action_indices

class DQNAgent:
    # Learning settings
    BATCH_SIZE = 8
    LR_INIT = 0.01
    LR_DECAY = 0.99975
    LR_MIN = 0.001
    DISCOUNT = 0.1 #gamma
    
    # Exploration settings
    EPSILON_INIT = 0.1
    EPSILON_DECAY = 0.99975
    EPSILON_MIN = 0.01
    
    CONV_UNITS = 1028
    DENSE_UNITS = 512
    
    def __init__(self, env):
        self.file_path = r'C:\Users\Admin\Documents'
        
        self.env = env
        self.iters = 0
        
        log_dir = os.path.join(self.file_path, 'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.tensorboard_callback = TensorBoard(log_dir=log_dir)

        self.discount = DQNAgent.DISCOUNT
        self.learn_rate = DQNAgent.LR_INIT
        self.epsilon = DQNAgent.EPSILON_INIT
        
        self.model = DQNAgent.__create_dqn(self.learn_rate,
                                           (*Environment.STATE_DIMS, 1),
                                           Environment.MAX_HEXES*2,)
            
        self.weight_path = os.path.join(self.file_path, 'logs', 'model_weights.h5')
        if os.path.isfile(self.weight_path):
            self.model.load_weights(self.weight_path)

        self.replay_memory = []

    @staticmethod
    def __create_dqn(learn_rate, input_dims, n_actions):
        model = Sequential([Conv2D(21, (3,3), activation='relu', padding='same', input_shape=input_dims),
                            Conv2D(14, (3,3), activation='relu', padding='same'),
                            Conv2D(7, (3,3), activation='relu', padding='same'),
                            Flatten(),
                            Dense(40, activation='relu'),
                            Dense(20, activation='relu'),
                            Dense(n_actions, activation='linear')])
    
        model.compile(optimizer=Adam(learning_rate=learn_rate, epsilon=1e-4), loss='mse')
        return model

    def get_action(self, state):
        unknown = self.env.unknown()

        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.unknown())
        else:
            moves = self.model.predict(np.reshape(state, (1, 7, 3, 1)))[0]
            temp = np.zeros_like(moves)-float('inf')
            temp[unknown] = moves[unknown]
            return np.argmax(temp)
            #move = np.argmax(moves)
            #print(moves.argsort()[-5:][::-1], moves[moves.argsort()[-5:][::-1]])

    def train(self, solved):
        if len(self.replay_memory) < DQNAgent.BATCH_SIZE:
            return
            
        np.random.shuffle(self.replay_memory)

        current_states = np.array([transition[0] for transition in self.replay_memory])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in self.replay_memory])
        future_qs_list = self.model.predict(new_current_states)

        X,y = [], []

        for i, (current_state, action, reward, new_current_state, done) in enumerate(self.replay_memory):
            if not solved:
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + DQNAgent.DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[i]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=DQNAgent.BATCH_SIZE,
                       shuffle=False, verbose=True,
                       callbacks=[self.tensorboard_callback])
        
        self.replay_memory = []
        self.iters += 1
        if solved or self.iters % 10 == 0:
            self.model.save_weights(self.weight_path)

        #self.learn_rate = max(DQNAgent.LR_MIN, self.learn_rate*DQNAgent.LR_DECAY)
        #self.epsilon = max(DQNAgent.EPSILON_MIN, self.epsilon*DQNAgent.EPSILON_DECAY)
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
def train():
    agent = DQNAgent(None)
    agent.env = Environment(GameParser(get_window()))
    
    solved = False
    while not solved:
        current_state = agent.env.get_state()

        action = agent.get_action(current_state)

        new_state, reward, solved = agent.env.step(action)
        
        agent.update_replay_memory((current_state, action, reward, new_state, solved))
        agent.train(solved)
            
"""
def test(model_path):
    env = Environment()
    agent = DQNAgent(env, model_path='model.h5')
    
    solved = False
    while not solved:
        action = agent.get_action(env.get_state())
        _, solved = env.step(action)
"""
if __name__ == '__main__': 
    menu = Navigator()
    menu.level_generator(train)
    
    