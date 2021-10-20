import os, pickle
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

from grid import Cell
from parse import GameParser
from window import get_window

class Environment:
    STATE_DIMS = (7, 3)
    MAX_HEXES = 10

    def _initial_state(self):
        state = np.zeros(Environment.STATE_DIMS)-1
        self._row_offset = 1 if self._grid[0, 1] is None else 0

        for row in range(self._grid.rows):
            for col in range(self._grid.cols):
                state[row+self._row_offset, col] = self._cell_to_rep(self._grid[row, col])

        return state

    def _cell_to_rep(self, cell):
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

    def _action_to_coords(self, action_index):
        return [(0,1), (1,0), (1,2),
                (2,1), (3,0), (3,2),
                (4,1), (5,0), (5,2),
                (6,1)][action_index]

    def _coords_to_action(self, coords):
        return [(0,1), (1,0), (1,2),
                (2,1), (3,0), (3,2),
                (4,1), (5,0), (5,2),
                (6,1)].index(coords)

    def get_state(self):
        return self._state

    def unknown(self):
        action_indices = []
        for cell in self._grid.unknown_cells():
            index = self._coords_to_action((cell.grid_coords[0]+self._row_offset, cell.grid_coords[1]))
            action_indices.append(index)
            action_indices.append(index+Environment.MAX_HEXES)

        return action_indices

class OfflineEnvironment(Environment):
    def __init__(self, grid, grid_solved):
        self._grid = grid
        self._grid_solved = grid_solved
        self._state = self._initial_state()

    def step(self, action_index):
        button = action_index // Environment.MAX_HEXES # Either left or right click
        action_index %= Environment.MAX_HEXES
        row, col = self._action_to_coords(action_index)

        reward = -1
        solved = False
        if row in range(Environment.STATE_DIMS[0]) and col in range(Environment.STATE_DIMS[1]):
            cell_curr = self._grid[row-self._row_offset, col]
            cell_true = self._grid_solved[row-self._row_offset, col]
            if cell_curr and cell_curr.colour == Cell.ORANGE:
                if ((button == 0 and cell_true.colour == Cell.BLUE) or
                    (button == 1 and cell_true.colour == Cell.BLACK)):

                    if cell_true.colour == Cell.BLUE:
                        self._grid.remaining -= 1

                    cell_curr.colour = cell_true.colour
                    if cell_true.digit is not None:
                        cell_curr.hint = cell_true.hint
                        cell_curr.digit = str(cell_true.digit)

                    self._state[row][col] = self._cell_to_rep(cell_true)

                    reward = 1
                    if len(self._grid.unknown_cells()) == 0:
                        solved = True

                    #print(self._grid)

        return self._state, reward, solved

class OnlineEnvironment(Environment):
    def __init__(self, parser):
        self.__parser = parser
        self._grid = self.__parser.parse_grid()
        self._state = self._initial_state()

    def step(self, action_index):
        button = action_index // Environment.MAX_HEXES # Either left or right click
        action_index %= Environment.MAX_HEXES
        row, col = self._action_to_coords(action_index)

        reward = -1
        solved = False
        if row in range(Environment.STATE_DIMS[0]) and col in range(Environment.STATE_DIMS[1]):
            cell = self._grid[row-self._row_offset, col]
            if cell and cell.colour == Cell.ORANGE:
                mistakes_old, _ = self.__parser.parse_counters()
                left_click_cells = [cell] if button == 0 else []
                right_click_cells = [cell] if button == 1 else []
                mistakes_new, remaining = self.__parser.parse_clicked(self._grid, left_click_cells, right_click_cells)

                if mistakes_old == mistakes_new:
                    reward = 1
                    if len(self._grid.unknown_cells()) == 0:
                        solved = True

        return self._state, reward, solved

class Agent:
    def __init__(self, environment=None, epsilon=0.5, discount=0.1, learning_rate=0.01, batch_size=64,
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

        self.__weights_path = os.path.join(self.__file_path, 'logs', 'model_weights.h5')
        if os.path.isfile(self.__weights_path):
            self.__model.load_weights(self.__weights_path)

    @property
    def environment(self):
        return self.__environment

    @environment.setter
    def environment(self, environment):
        self.__environment = environment

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
            action_rewards = self.__model.predict(np.reshape(state, (1, *Environment.STATE_DIMS, 1)))[0]
            temp = np.zeros_like(action_rewards)-float('inf')
            temp[unknown] = action_rewards[unknown]
            return np.argmax(temp)
            #print(moves.argsort()[-5:][::-1], moves[moves.argsort()[-5:][::-1]])

    def train(self):
        if len(self.__replay_memory) < self.__batch_size:
            return

        np.random.shuffle(self.__replay_memory)

        current_states = np.array([transition[0] for transition in self.__replay_memory])
        current_state_rewards = self.__model.predict(np.expand_dims(current_states, -1))

        new_states = np.array([transition[3] for transition in self.__replay_memory])
        new_state_rewards = self.__model.predict(np.expand_dims(new_states, -1))

        X, y = [], []
        for i, (current_state, action, reward, new_current_state, solved) in enumerate(self.__replay_memory):
            if solved:
                new_reward = reward
            else:
                new_reward = reward + self.__discount*np.max(new_state_rewards[i])

            current_state_rewards[i][action] = new_reward

            X.append(np.expand_dims(current_state, -1))
            y.append(current_state_rewards[i])

        self.__model.fit(np.array(X), np.array(y), batch_size=self.__batch_size,
                         shuffle=False, verbose=False)

    def update_replay_memory(self, transition):
        if len(self.__replay_memory) >= self.__max_replay_memory:
            self.__replay_memory.pop(np.random.randint(0, len(self.__replay_memory)))

        self.__replay_memory.append(transition)

    def save_weights(self):
        self.__model.save_weights(self.__weights_path)

def train(epochs=5, levels_path='resources/levels/levels.pickle'):
    agent = Agent()
    for epoch in range(epochs):
        print('Epoch {}'.format(epoch+1))
        with open(levels_path, 'rb') as file:
            levels, _ = pickle.load(file)

        for i, (grid, grid_solved) in enumerate(levels, 1):
            agent.environment = environment = OfflineEnvironment(grid, grid_solved)

            solved = False
            while not solved:
                current_state = environment.get_state()

                action = agent.get_action(current_state)

                new_state, reward, solved = environment.step(action)

                agent.update_replay_memory((current_state, action, reward, new_state, solved))
                agent.train()

            agent.save_weights()
            if i % 10 == 0:
                print('>>> {0}/{1}'.format(i, len(levels)))

        print()

def test_offline(levels_path='resources/levels/levels.pickle'):
    with open(levels_path, 'rb') as file:
        levels, _ = pickle.load(file)

    agent = Agent()
    for i, (grid, grid_solved) in enumerate(levels):
        agent.environment = environment = OfflineEnvironment(grid, grid_solved)

        solved = False
        mistakes = 0
        while not solved:
            current_state = environment.get_state()
            action = agent.get_action(current_state)
            _, reward, solved = environment.step(action)
            if reward == -1:
                mistakes += 1

        print('>>> {0}/{1} - Mistakes: {2}'.format(i+1, len(levels), mistakes))

def test_online():
    environment = OnlineEnvironment(GameParser(get_window()))
    agent = Agent(environment)

    solved = False
    while not solved:
        current_state = environment.get_state()
        action = agent.get_action(current_state)
        _, _, solved = environment.step(action)

if __name__ == '__main__':
    train()
    #test_offline()
    #test_online()
