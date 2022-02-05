import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import pickle
import numpy as np

from abc import ABC, abstractmethod

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

from grid import Cell
from solve import Solver


class Environment(ABC):
    STATE_DIMS = (7, 3)
    MAX_HEXES = 10

    @abstractmethod
    def __init__(self):
        self._grid = None
        self._state = None
        raise NotImplementedError

    def get_state(self):
        return self._state

    def _initial_state(self):
        state = np.zeros(Environment.STATE_DIMS)
        self._row_offset = 1 if self._grid[0, 1] is None else 0

        for row in range(self._grid.rows):
            for col in range(self._grid.cols):
                state[row + self._row_offset, col] = Environment._cell_to_rep(self._grid[row, col])

        return state

    @staticmethod
    def _cell_to_rep(cell):
        if cell is None:
            return 0

        elif cell.colour == Cell.BLACK:
            if cell.number == '?':
                return 0.8
            else:
                return (cell.number + 1) / 10

        elif cell.colour == Cell.BLUE:
            return 0.9

        elif cell.colour == Cell.ORANGE:
            return 1.0

    @staticmethod
    def _action_to_coords(action_index):
        return [(0, 1), (1, 0), (1, 2),
                (2, 1), (3, 0), (3, 2),
                (4, 1), (5, 0), (5, 2),
                (6, 1)][action_index]

    @staticmethod
    def _coords_to_action(coords):
        return [(0, 1), (1, 0), (1, 2),
                (2, 1), (3, 0), (3, 2),
                (4, 1), (5, 0), (5, 2),
                (6, 1)].index(coords)

    def unknown(self):
        action_indices = []
        for cell in self._grid.unknown_cells():
            index = self._coords_to_action((cell.grid_coords[0] + self._row_offset, cell.grid_coords[1]))
            action_indices.append(index)
            action_indices.append(index + Environment.MAX_HEXES)

        return action_indices


class OfflineEnvironment(Environment):
    def __init__(self, grid, grid_solved):
        self._grid = grid
        self._grid_solved = grid_solved
        self._state = self._initial_state()

    def step(self, agent_action):
        solver = Solver(None)
        left_click_cells, right_click_cells = solver.solve_single_step(self._grid)
        left_click_coords = [cell.grid_coords for cell in left_click_cells]
        right_click_coords = [cell.grid_coords for cell in right_click_cells]

        rewards = []
        solved = False
        for action in range(2 * Environment.MAX_HEXES):
            button = action // Environment.MAX_HEXES  # Either left or right click
            row, col = Environment._action_to_coords(action % Environment.MAX_HEXES)

            if (((row - self._row_offset, col) in left_click_coords and button == 0) or
                    ((row - self._row_offset, col) in right_click_coords and button == 1)):
                rewards.append(1)
            else:
                rewards.append(-1)

            if action == agent_action:
                cell_curr = self._grid[row - self._row_offset, col]
                cell_true = self._grid_solved[row - self._row_offset, col]

                if cell_curr and cell_curr.colour == Cell.ORANGE:
                    if cell_true.colour == Cell.BLUE:
                        self._grid.remaining -= 1

                    cell_curr.colour = cell_true.colour
                    if cell_true.number is not None:
                        cell_curr.hint = cell_true.hint
                        cell_curr.number = str(cell_true.number)

                    self._state[row][col] = Environment._cell_to_rep(cell_true)
                    if len(self._grid.unknown_cells()) == 0:
                        solved = True

        return self._state, rewards, solved


"""
class OnlineEnvironment(Environment):
    def __init__(self, parser):
        self.__parser = parser
        self._grid = self.__parser.parse_grid()
        self._state = self._initial_state()

    def step(self, action_index):
        button = action_index // Environment.MAX_HEXES # Either left or right click
        action_index %= Environment.MAX_HEXES
        row, col = Environment._action_to_coords(action_index)

        reward = -2
        solved = False
        if row in range(Environment.STATE_DIMS[0]) and col in range(Environment.STATE_DIMS[1]):
            cell = self._grid[row-self._row_offset, col]
            if cell and cell.colour == Cell.ORANGE:
                mistakes_old, _ = self.__parser.parse_counters()
                left_click_cells = [cell] if button == 0 else []
                right_click_cells = [cell] if button == 1 else []
                mistakes_new, remaining = self.__parser.parse_clicked(self._grid, left_click_cells, right_click_cells)

                self._grid.remaining = remaining

                if mistakes_old == mistakes_new:
                    reward = 1
                    if len(self._grid.unknown_cells()) == 0:
                        solved = True

        return self._state, reward, solved
"""


class Agent:
    def __init__(self, environment, batch_size, learning_rate, discount_rate, exploration_rate, max_replay_memory,
                 replay=True, double=False, target_update_interval=5, model_path=None, save_path=r'resources/models'):
        self.__environment = environment
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__discount_rate = discount_rate
        self.__exploration_rate = exploration_rate
        
        self.__replay_memory = []
        self.__max_replay_memory = max_replay_memory if replay else 1

        if model_path:
            if os.path.isfile(model_path):
                self.__model = load_model(model_path)
                self.__save_path = model_path
            else:
                raise FileNotFoundError
        else:
            self.__model = self.__create_model((*Environment.STATE_DIMS, 1), Environment.MAX_HEXES * 2)
            
            i = 1
            while True:
                filename = f'model_{i}'
                if not os.path.isfile(os.path.join(save_path, filename)):
                    break
                i += 1
            
            self.__save_path = os.path.join(save_path, filename)
        
        if double:
            self.__target_model = self.__create_model((*Environment.STATE_DIMS, 1), Environment.MAX_HEXES * 2)
            self.__target_model.set_weights(self.__model.get_weights())
            self.__target_update_interval = target_update_interval
            self.__target_update_counter = 0
        else:
            self.__target_model = None

    @property
    def environment(self):
        return self.__environment

    @environment.setter
    def environment(self, environment):
        self.__environment = environment

    def __create_model(self, input_dims, num_actions):
        model = Sequential([Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_dims),
                            Conv2D(128, (3, 3), activation='relu', padding='same'),
                            Conv2D(128, (3, 3), activation='relu', padding='same'),
                            Conv2D(128, (3, 3), activation='relu', padding='same'),
                            Flatten(),
                            Dense(512, activation='relu'),
                            Dense(512, activation='relu'),
                            Dense(num_actions, activation='linear')])

        model.compile(optimizer=Adam(learning_rate=self.__learning_rate), loss='mse')
        return model

    def get_action(self, state, testing=False):
        unknown = self.__environment.unknown()

        if not testing and np.random.random() < self.__exploration_rate:
            return np.random.choice(self.__environment.unknown())
        else:
            action_rewards = self.__model.predict(np.reshape(state, (1, *Environment.STATE_DIMS, 1)))[0]
            temp = np.zeros_like(action_rewards) - float('inf')
            temp[unknown] = action_rewards[unknown]
            return np.argmax(temp)

    def train(self, transition):
        if len(self.__replay_memory) >= self.__max_replay_memory:
            self.__replay_memory.pop(np.random.randint(0, len(self.__replay_memory)))

        self.__replay_memory.append(transition)

        batch = random.sample(self.__replay_memory, min(self.__batch_size, len(self.__replay_memory)))

        current_states = np.array([np.expand_dims(s, -1) for s, _, _, _, _ in batch])
        current_state_rewards = np.array([r for _, _, r, _, _ in batch])

        predictor = self.__target_model if self.__target_model else self.__model
        next_states = np.array([np.expand_dims(s, -1) for _, _, _, s, _ in batch])
        next_state_rewards = predictor.predict(np.expand_dims(next_states, -1))
            
        for i, (_, action, _, _, solved) in enumerate(batch):
           if not solved:
               current_state_rewards[i][action] += self.__discount_rate*np.max(next_state_rewards[i])
           
        self.__model.fit(current_states, np.array(current_state_rewards),
                         batch_size=self.__batch_size, shuffle=False, verbose=False)
        
        if self.__target_model:
            # If solved
            if transition[4]:
                self.__target_update_counter += 1
                
            if self.__target_update_counter > self.__target_update_interval:
                self.__target_model.set_weights(self.__model.get_weights())
                self.__target_update_counter = 0
               
        # Decrease epsilion

    def save_model(self):
        self.__model.save(self.__save_path)


class Trainer:
    @staticmethod
    def __load_levels(num_training=float('inf'), num_testing=float('inf'),
                      levels_path='resources/levels/levels.pickle'):
        with open(levels_path, 'rb') as file:
            levels, _ = pickle.load(file)

            if num_training == float('inf'):
                train_levels = levels
            else:
                assert num_training <= len(levels)
                train_levels = levels[:num_training]

            if num_testing == float('inf'):
                test_levels = levels
            else:
                assert num_training + num_testing <= len(levels)
                test_levels = levels[num_training:num_training + num_testing]

        return train_levels, test_levels

    @staticmethod
    def __train_test_accuracy(agent, num_train, num_test):
        train_levels, test_levels = Trainer.__load_levels(num_train, num_test)
        train_accuracy = Trainer.__accuracy_offline(agent, train_levels)
        test_accuracy = Trainer.__accuracy_offline(agent, test_levels)
        print(f'Training Accuracy: {train_accuracy}')
        print(f'Test Accuracy: {test_accuracy}\n')

    @staticmethod
    def __accuracy_offline(agent, levels):
        actions = 0
        mistakes = 0
        for grid, grid_solved in levels:
            agent.environment = environment = OfflineEnvironment(grid, grid_solved)

            solved = False
            while not solved:
                current_state = environment.get_state()
                action = agent.get_action(current_state, testing=True)
                _, rewards, solved = environment.step(action)

                actions += 1
                if rewards[action] != 1:
                    mistakes += 1

        return 1 - (mistakes / actions)

    @staticmethod
    def train_offline(test_only=False, epochs=50, batch_size=64, learning_rate=0.01, discount_rate=0.05,
                      exploration_rate=0, max_replay_memory=10000, replay=True,
                      double=False, target_update_interval=5, model_path=None):
        num_train = 100
        num_test = 20

        agent = Agent(None, batch_size, learning_rate, discount_rate, exploration_rate,
                      max_replay_memory, replay, double, target_update_interval, model_path)
        Trainer.__train_test_accuracy(agent, num_train, num_test)

        for epoch in range(epochs):
            print('Epoch {}'.format(epoch + 1))
            train_levels, _ = Trainer.__load_levels(num_train, 0)

            for i, (grid, grid_solved) in enumerate(train_levels, 1):
                agent.environment = environment = OfflineEnvironment(grid, grid_solved)

                solved = False
                while not solved:
                    current_state = environment.get_state()
                    action = agent.get_action(current_state)
                    new_state, rewards, solved = environment.step(action)

                    if not test_only:
                        agent.train((current_state, action, rewards, new_state, solved))

                if i % 100 == 0:
                    print('>>> {0}/{1}'.format(i, len(train_levels)))
                    agent.save_model()

            agent.save_model()
            Trainer.__train_test_accuracy(agent, num_train, num_test)

    @staticmethod
    def test_online():
        pass


if __name__ == '__main__':
    Trainer.train_offline()
    # Trainer.train_online()
