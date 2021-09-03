import numpy as np
import os

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam

from solve import Solver

from grid import Cell
import datetime

from keras.callbacks import TensorBoard

class Environment:
    STATE_DIMS = (33,33)
    MAX_HEXES = (STATE_DIMS[0]-1)*(STATE_DIMS[1])-1
    
    def __init__(self, parser):
        self.__parser = parser
        self.__solver =  Solver(parser)
        self.__grid = self.__parser.parse_grid()
        self.__state = self.__get_state()
    
    def get_state(self):
        return self.__state
    
    def __get_state(self):
        state = np.zeros((self.__grid.rows, self.__grid.cols, 3))
        
        for row in range(self.__grid.rows):
            for col in range(self.__grid.cols):
                cell = self.__grid[row, col]
                rep = [0,0,0]
                if cell:
                    rep[0] = Cell.COLOURS.index(cell.colour)+1
                    if cell.digit:
                        rep[1] = cell.digit if cell.digit != '?' else 0
                    if cell.hint:
                        rep[2] = Cell.HINT_TYPES.index(cell.hint)
                    
                state[row, col] = rep
        
        state_padded = np.zeros((*Environment.STATE_DIMS, 3))
        state_padded[1:self.__grid.rows+1, 1:self.__grid.cols+1] = state
        
        for constraint in self.__grid.constraints:
            rep = [0,0,0]
            rep[0] = constraint.angle
            rep[1] = constraint.size
            rep[2] = Cell.HINT_TYPES.index(constraint.hint)
            
            row, col = constraint.members[0].grid_coords
            if constraint.angle == 0:
                row -= 1
                
            elif constraint.angle == 60:
                row -= 1
                col += 1
            
            elif constraint.angle == 300:
                row -= 1
                col -= 1
                
            else:
                raise NotImplementedError('DQN agent only supports constraints at angles 0, 60 and 300 degrees')
            
            state_padded[row+1, col+1] = rep
        
        state_padded[-1, -1] = self.__grid.remaining
        
        return state_padded
        
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
                    
                    remaining = self.__parser.parse_clicked(self.__grid, left_click_cells, right_click_cells)
                    if remaining:
                        self.__grid.remaining = self.__state[-1, -1] = remaining
                    
                    rep = [0,0,0]
                    rep[0] = Cell.COLOURS.index(cell.colour)+1
                    if cell.digit:
                        rep[1] = cell.digit if cell.digit != '?' else 0
                    if cell.hint:
                        rep[2] = Cell.HINT_TYPES.index(cell.hint)
                    
                    self.__state[row+1, col+1] = rep
                    reward = 1
        
        print(reward)
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
    EPSILON_INIT = 0.5
    EPSILON_DECAY = 0.99975
    EPSILON_MIN = 0.01
    
    CONV_UNITS = 1028
    DENSE_UNITS = 512
    
    def __init__(self, env):
        self.file_path = r'C:\Users\james\Documents'
        
        self.env = env
        self.iters = 0
        
        log_dir = os.path.join(self.file_path, 'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.tensorboard_callback = TensorBoard(log_dir=log_dir)

        self.discount = DQNAgent.DISCOUNT
        self.learn_rate = DQNAgent.LR_INIT
        self.epsilon = DQNAgent.EPSILON_INIT
        
        self.model = DQNAgent.__create_dqn(self.learn_rate,
                                           (*Environment.STATE_DIMS, 3),
                                           Environment.MAX_HEXES*2,
                                           DQNAgent.CONV_UNITS, 
                                           DQNAgent.DENSE_UNITS)
            
        self.weight_path = os.path.join(self.file_path, 'logs', 'model_weights.h5')
        if os.path.isfile(self.weight_path):
            self.model.load_weights(self.weight_path)

        self.replay_memory = []

    @staticmethod
    def __create_dqn(learn_rate, input_dims, n_actions, conv_units, dense_units):
        model = Sequential([Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_dims),
                            Conv2D(64, (3,3), activation='relu', padding='same'),
                            Conv2D(64, (3,3), activation='relu', padding='same'),
                            Conv2D(64, (3,3), activation='relu', padding='same'),
                            Flatten(),
                            Dense(128, activation='relu'),
                            Dense(128, activation='relu'),
                            Dense(n_actions, activation='linear')])
    
        model.compile(optimizer=Adam(lr=learn_rate, epsilon=1e-4), loss='mse')
        return model

    def get_action(self, state):
        unknown = self.env.unknown()

        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.unknown())
        else:
            moves = self.model.predict(np.expand_dims(state, axis=0))[0]
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
        
def train(agent):
    for _ in range(1):
        episode_reward = 0
        solved = False
        while not solved:
            current_state = agent.env.get_state()

            action = agent.get_action(current_state)

            new_state, reward, solved = agent.env.step(action)

            episode_reward += reward
            
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
    from navigate import Navigator
    from parse import GameParser, MenuParser
    from window import get_window
    
    import time
    
    menu = Navigator()
    
    window = get_window()
    menu_parser = MenuParser(window)
    
    agent = DQNAgent(None)
  
    for level in ['1-2', '1-6']:
        menu.load_level(level)
        parser = GameParser(window)
        env = Environment(parser)
        agent.env = env
        train(agent)
        
        time.sleep(2)
        next_button, menu_button = menu_parser.parse_level_end()
        window.click(menu_button)
        time.sleep(3)
    