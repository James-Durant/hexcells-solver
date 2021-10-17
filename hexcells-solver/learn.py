import os, pickle, datetime
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from grid import Cell

class Environment:
    STATE_DIMS = (7, 3)
    MAX_HEXES = 10
    
    def __init__(self, grid, grid_solved):
        self.__grid = grid
        self.__grid_solved = grid_solved
        self.__state = self.__initial_state()
    
    def __initial_state(self):
        return [[self.__cell_to_rep(self.__grid[row, col]) for col in range(Environment.STATE_DIMS[0])]
                for row in range(Environment.STATE_DIMS[1])]
    
    def __cell_to_rep(self, cell):
        if cell is None:
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
            
        return rep
    
    def __action_to_coords(self, action_index):
        return [(0,1), (1,0), (1,2),
                (2,1), (3,0), (3,2),
                (4,1), (5,0), (5,2),
                (6,1)][action_index]
    
    def __coords_to_action(self, coords):
        return [(0,1), (1,0), (1,2),
                (2,1), (3,0), (3,2),
                (4,1), (5,0), (5,2),
                (6,1)].index(coords)
    
    def get_state(self):
        return self.__state
    
    def step(self, action_index):
        button = action_index // Environment.MAX_HEXES # Either left or right click
        action_index %= Environment.MAX_HEXES
        row, col = self.__action_to_coords(action_index)
        
        reward = -1
        solved = False
        if row in range(Environment.STATE_DIMS[0]) and col in range(Environment.STATE_DIMS[1]):
            cell_curr = self.__grid[row, col]
            cell_true = self.__grid_solved[row, col]
            if cell_curr and cell_curr.colour == Cell.ORANGE:
                if ((button == 0 and cell_true.colour == Cell.BLUE) or
                    (button == 1 and cell_true.colour == Cell.BLACK)):
                    self.__grid[row, col] = cell_true
                    if cell_true.colour == Cell.BLUE:
                        self.__grid.remaining -= 1
                    
                    self.__state[row][col] = self.__cell_to_rep(cell_true)
                    
                    reward = 1       
                    if len(self.__grid.unknown_cells() == 0):
                        solved = True
                        
        return self.__state, reward, solved
    
    def unknown(self):
        action_indices = []
        for cell in self.__grid.unknown_cells():
            index = self.__coords_to_action(cell.grid_coords)
            action_indices.append(index)
            action_indices.append(index+Environment.MAX_HEXES)
            
        return action_indices

class Agent:
    def __init__(self, environment, epsilon=0.1, discount=0.1, learning_rate=0.01, batch_size=64,
                 max_replay_memory=1024, file_path=r'C:\Users\Admin\Documents'):
        self.__environment = environment
        self.__epsilon = epsilon
        self.__discount = discount
        self.__learning_rate = learning_rate
        self.__batch_size = batch_size
        self.__file_path = file_path
        self.__max_replay_memory = max_replay_memory
        self.__replay_memory = []
        
        self.__model = self.__create_model((*Environment.STATE_DIMS, 1), Environment.MAX_HEXES*2)
            
        log_dir = os.path.join(self.__file_path, 'logs', 'fit', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.__tensorboard_callback = TensorBoard(log_dir=log_dir)
        
        self.__weights_path = os.path.join(self.__file_path, 'logs', 'model_weights.h5')
        if os.path.isfile(self.__weights_path):
            self.__model.load_weights(self.__weights_path)

    def __create_model(self, input_dims, num_actions):
        model = Sequential([Conv2D(21, (3,3), activation='relu', padding='same', input_shape=input_dims),
                            Conv2D(14, (3,3), activation='relu', padding='same'),
                            Conv2D(7, (3,3), activation='relu', padding='same'),
                            Flatten(),
                            Dense(40, activation='relu'),
                            Dense(20, activation='relu'),
                            Dense(num_actions, activation='linear')])
    
        model.compile(optimizer=Adam(learning_rate=self.__learning_rate, epsilon=1e-4), loss='mse')
        return model

    def get_action(self, state):
        unknown = self.__environment.unknown()

        if np.random.random() < self.__epsilon:
            return np.random.choice(self.__environment.unknown())
        else:
            action_rewards = self.__model.predict(np.reshape(state, (1, 7, 3, 1)))[0]
            temp = np.zeros_like(action_rewards)-float('inf')
            temp[unknown] = action_rewards[unknown]
            return np.argmax(temp)
            #print(moves.argsort()[-5:][::-1], moves[moves.argsort()[-5:][::-1]])

    def train(self):
        if len(self.__replay_memory) < self.__batch_size:
            return
            
        np.random.shuffle(self.__replay_memory)

        current_states = np.array([transition[0] for transition in self.__replay_memory])
        current_state_rewards = self.__model.predict(current_states)

        new_states = np.array([transition[3] for transition in self.__replay_memory])
        new_state_rewards = self.__model.predict(new_states)

        X, y = [], []
        for i, (current_state, action, reward, new_current_state, solved) in enumerate(self.__replay_memory):
            if solved:
                new_reward = reward
            else:
                new_reward = reward + self.__discount*np.max(new_state_rewards[i])

            current_state_rewards[i][action] = new_reward

            X.append(current_state)
            y.append(current_state_rewards[i])

        self.__model.fit(np.array(X), np.array(y), batch_size=self.__batch_size,
                         shuffle=False, verbose=True, callbacks=[self.__tensorboard_callback])
    
    def update_replay_memory(self, transition):
        if len(self.__replay_memory) >= self.__max_replay_memory:
            self.__replay_memory = list(np.random.choice(self.__replay_memory, self.__max_replay_memory-1, replace=False))

        self.__replay_memory.append(transition)
        
    def save_weights(self):
        self.__model.save_weights(self.__weights_path)
        
def train(levels_path='resources/levels/levels.pickle'):
    with open(levels_path, 'rb') as file:
        levels, _ = pickle.load(file)
    
    for i, (grid, grid_solved) in enumerate(levels):
        environment = Environment(grid, grid_solved)    
        agent = Agent(environment)

        solved = False
        while not solved:
            current_state = environment.get_state()

            action = agent.get_action(current_state)

            new_state, reward, solved = environment.step(action)
        
            agent.update_replay_memory((current_state, action, reward, new_state, solved))
            agent.train()
            
        agent.save_weights()   
        print('>>> {0}/{1}'.format(i+1, len(levels)))
            
if __name__ == '__main__': 
    train()
    