import os
import time
import random
import pickle
import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

from grid import Cell
from solve import Solver
from parse import LevelParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Default values
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 0.99975
LEARNING_RATE_MIN = 0.00001
DISCOUNT_RATE = 0
EXPLORATION_RATE = 0.95
EXPLORATION_RATE_DECAY = 0.99975
EXPLORATION_RATE_MIN = 0.01
EXPERIENCE_REPLAY = False
MAX_REPLAY_MEMORY = 50000
DOUBLE_DQN = False
TARGET_UPDATE_INTERVAL = 5
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources/models')


class Environment:
    def __init__(self, grid):
        self._grid = grid
        self._offset = 1 if self._grid[0, 0] is None else 0
        self._state_dims = (grid.rows, grid.cols)
        self._max_cells = (grid.rows * grid.cols) // 2
        self._state = self._initial_state()

    @property
    def state_dims(self):
        return self._state_dims

    @property
    def max_cells(self):
        return self._max_cells

    def get_state(self):
        return self._state

    def _initial_state(self):
        state = np.zeros(self._state_dims)

        for row in range(self._grid.rows):
            for col in range(self._grid.cols):
                state[row, col] = Environment._cell_to_rep(self._grid[row, col])

        return state

    @staticmethod
    def _cell_to_rep(cell):
        if cell is None:
            return 0.0

        elif cell.colour == Cell.BLACK:
            if cell.number == '?':
                return 0.8
            else:
                return (cell.number + 1) / 10

        elif cell.colour == Cell.BLUE:
            return 0.9

        elif cell.colour == Cell.ORANGE:
            return 1.0

    def _action_to_coords(self, action_index):
        row = (2 * action_index + self._offset) // self._grid.cols
        col = (2 * action_index + self._offset) % self._grid.cols
        return int(row), int(col)

    def _coords_to_action(self, coords):
        return int((coords[0] * self._grid.cols + coords[1]) // 2)

    def unknown(self):
        action_indices = []
        for cell in self._grid.unknown_cells():
            index = self._coords_to_action(cell.grid_coords)
            action_indices.append(index)
            action_indices.append(index + self._max_cells)

        return action_indices

    def _reward(self, action, left_click_coords, right_click_coords):
        button = action // self._max_cells  # Either left or right click
        row, col = self._action_to_coords(action % self._max_cells)

        if ((row, col) in left_click_coords and not button) or ((row, col) in right_click_coords and button):
            return button, row, col, 1
        else:
            return button, row, col, -1


class OfflineEnvironment(Environment):
    def __init__(self, grid, grid_solved):
        super().__init__(grid)
        self._grid_solved = grid_solved

    def step(self, agent_action):
        solver = Solver(None)
        left_click_cells, right_click_cells = solver.solve_single_step(self._grid)
        left_click_coords = [cell.grid_coords for cell in left_click_cells]
        right_click_coords = [cell.grid_coords for cell in right_click_cells]

        rewards = []
        solved = False
        for action in range(2 * self._max_cells):
            button, row, col, reward = self._reward(action, left_click_coords, right_click_coords)
            rewards.append(reward)

            if action == agent_action:
                cell_curr = self._grid[row, col]
                cell_true = self._grid_solved[row, col]

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


class OnlineEnvironment(Environment):
    def __init__(self, parser, delay):
        super().__init__(parser.parse_grid())
        self.__parser = parser
        self.__delay = delay

    def step(self, agent_action):
        solver = Solver(self.__parser)
        left_click_cells, right_click_cells = solver.solve_single_step(self._grid)
        left_click_coords = [cell.grid_coords for cell in left_click_cells]
        right_click_coords = [cell.grid_coords for cell in right_click_cells]

        rewards = []
        solved = False
        for action in range(2 * self._max_cells):
            button, row, col, reward = self._reward(action, left_click_coords, right_click_coords)
            rewards.append(reward)

            if action == agent_action:
                cell = self._grid[row, col]

                if (row, col) in left_click_coords:
                    if rewards[-1] == -1:
                        self.__parser.click_cells([cell], 'right')
                        time.sleep(0.1)

                    if len(self._grid.unknown_cells()) > 1:
                        _, remaining = self.__parser.parse_clicked(self._grid, [cell], [], self.__delay)
                    else:
                        self.__parser.click_cells([cell], 'left')
                        cell.colour = Cell.BLUE
                        remaining = 0
                        solved = True

                elif (row, col) in right_click_coords:
                    if rewards[-1] == -1:
                        self.__parser.click_cells([cell], 'left')
                        time.sleep(0.1)

                    if len(self._grid.unknown_cells()) > 1:
                        _, remaining = self.__parser.parse_clicked(self._grid, [], [cell], self.__delay)
                    else:
                        self.__parser.click_cells([cell], 'right')
                        cell.colour = Cell.BLACK
                        cell.number = '?'
                        remaining = 0
                        solved = True

                else:
                    mistakes_before, remaining = self.__parser.parse_counters()

                    if button == 0:
                        mistakes_after, remaining = self.__parser.parse_clicked(self._grid, [cell], [], self.__delay)
                        if mistakes_after > mistakes_before:
                            time.sleep(0.1)
                            _, remaining = self.__parser.parse_clicked(self._grid, [], [cell], self.__delay)

                    elif button == 1:
                        mistakes_after, remaining = self.__parser.parse_clicked(self._grid, [], [cell], self.__delay)
                        if mistakes_after > mistakes_before:
                            time.sleep(0.1)
                            _, remaining = self.__parser.parse_clicked(self._grid, [cell], [], self.__delay)

                self._grid.remaining = remaining
                self._state[row][col] = Environment._cell_to_rep(cell)

        return self._state, rewards, solved


class Agent:
    def __init__(self, environment, batch_size, learning_rate, discount_rate, exploration_rate,
                 experience_replay=EXPERIENCE_REPLAY, double_dqn=DOUBLE_DQN,
                 target_update_interval=TARGET_UPDATE_INTERVAL,
                 model_path=None, save_path=SAVE_PATH):
        self.__environment = environment
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__discount_rate = discount_rate
        self.__exploration_rate = exploration_rate

        self.__replay_memory = []
        self.__max_replay_memory = MAX_REPLAY_MEMORY if experience_replay else 1

        if model_path is not None:
            if os.path.isfile(model_path):
                self.__model = load_model(model_path)
                self.__model_path = model_path
            else:
                raise FileNotFoundError
        else:
            self.__model = self.__create_model((*self.__environment.state_dims, 1), self.__environment.max_cells * 2)

            i = 1
            while True:
                filename = f'model_{i}.h5'
                if not os.path.isfile(os.path.join(save_path, filename)):
                    break
                i += 1

            self.__model_path = os.path.join(save_path, filename)

        self.save_model()

        if double_dqn:
            self.__target_model = self.__create_model((*self.__environment.state_dims, 1),
                                                      self.__environment.max_cells * 2)
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

    @property
    def model_path(self):
        return self.__model_path

    def __create_model(self, input_dims, num_actions):
        model = Sequential([Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_dims),
                            Conv2D(64, (3, 3), activation='relu', padding='same'),
                            Conv2D(64, (3, 3), activation='relu', padding='same'),
                            Conv2D(64, (3, 3), activation='relu', padding='same'),
                            Flatten(),
                            Dense(256, activation='relu'),
                            Dense(256, activation='relu'),
                            Dense(256, activation='relu'),
                            Dense(256, activation='relu'),
                            Dense(num_actions, activation='linear')])

        model.compile(optimizer=Adam(learning_rate=self.__learning_rate), loss='mse')
        return model

    def get_action(self, state, test_only=False):
        unknown = self.__environment.unknown()

        if not test_only and np.random.random() < self.__exploration_rate:
            return np.random.choice(self.__environment.unknown())
        else:
            action_rewards = self.__model.predict(np.reshape(state, (1, *self.__environment.state_dims, 1)))[0]
            filtered_rewards = np.zeros_like(action_rewards) - float('inf')
            filtered_rewards[unknown] = action_rewards[unknown]
            return np.argmax(filtered_rewards)

    def train(self, transition):
        if len(self.__replay_memory) >= self.__max_replay_memory:
            self.__replay_memory.pop(np.random.randint(0, len(self.__replay_memory)))

        self.__replay_memory.append(transition)

        batch = random.sample(self.__replay_memory, min(self.__batch_size, len(self.__replay_memory)))

        current_states = np.array([np.expand_dims(s, -1) for s, _, _, _, _ in batch])
        current_state_rewards = np.array([r for _, _, r, _, _ in batch])

        if self.__discount_rate > 0:
            predictor = self.__target_model if self.__target_model else self.__model
            next_states = np.array([np.expand_dims(s, -1) for _, _, _, s, _ in batch])
            next_state_rewards = predictor.predict(next_states)

            for i, (_, action, _, _, solved) in enumerate(batch):
                if not solved:
                    current_state_rewards[i][action] += self.__discount_rate * np.max(next_state_rewards[i])

        self.__model.fit(current_states, np.array(current_state_rewards), shuffle=False, verbose=False)

        if self.__target_model:
            # If solved
            if transition[4]:
                self.__target_update_counter += 1

            if self.__target_update_counter > self.__target_update_interval:
                self.__target_model.set_weights(self.__model.get_weights())
                self.__target_update_counter = 0

        self.__learning_rate = max(LEARNING_RATE_MIN, self.__learning_rate * LEARNING_RATE_DECAY)
        self.__exploration_rate = max(EXPLORATION_RATE_MIN, self.__exploration_rate * EXPLORATION_RATE_DECAY)

    def save_model(self):
        self.__model.save(self.__model_path)


class Trainer:
    @staticmethod
    def __load_levels(levels_path, test_train_split):
        with open(levels_path, 'rb') as file:
            levels, _ = pickle.load(file)

        split = round(len(levels) * test_train_split)
        return levels[:split], levels[split:]

    @staticmethod
    def __train_test_accuracy(agent, levels_path, test_train_split, level_count, new_log=False):
        print('Computing accuracy...')
        train_levels, test_levels = Trainer.__load_levels(levels_path, test_train_split)
        train_accuracy = Trainer.__accuracy_offline(agent, train_levels)
        test_accuracy = Trainer.__accuracy_offline(agent, test_levels)
        print(f'Training Accuracy: {train_accuracy}')
        print(f'Test Accuracy: {test_accuracy}\n')

        save_path, filename = os.path.split(agent.model_path)
        save_path = os.path.join(save_path, 'logs')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        filename = f'{os.path.splitext(filename)[0]}.csv'
        file_path = os.path.join(save_path, filename)
        write_mode = 'w' if new_log or not os.path.isfile(file_path) else 'a'

        with open(file_path, write_mode) as file:
            file.write(f'{level_count}, {train_accuracy}, {test_accuracy}\n')

    @staticmethod
    def __accuracy_offline(agent, levels):
        if len(levels) == 0:
            raise RuntimeWarning('Calculating accuracy with no levels')

        actions = 0
        mistakes = 0
        for i, (grid, grid_solved) in enumerate(levels, 1):
            agent.environment = environment = OfflineEnvironment(grid, grid_solved)

            solved = False
            while not solved:
                current_state = environment.get_state()
                action = agent.get_action(current_state, test_only=True)
                _, rewards, solved = environment.step(action)

                actions += 1
                if rewards[action] != 1:
                    mistakes += 1

            if i % 100 == 0:
                print(f'>>> {i}/{len(levels)}')

        return 1 - (mistakes / actions)

    @staticmethod
    def train_offline(epochs, test_only=False, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                      discount_rate=DISCOUNT_RATE, exploration_rate=EXPLORATION_RATE,
                      experience_replay=EXPERIENCE_REPLAY, double_dqn=DOUBLE_DQN, model_path=None, level_size='large',
                      test_train_split=0.8):

        current_path = os.path.dirname(os.path.abspath(__file__))
        level_path = os.path.join(current_path, f'resources/levels/levels_{level_size}.pickle')

        train_levels, _ = Trainer.__load_levels(level_path, 1)
        environment = OfflineEnvironment(*train_levels[0])

        agent = Agent(environment, batch_size, learning_rate, discount_rate, exploration_rate,
                      experience_replay, double_dqn, model_path=model_path)

        Trainer.__train_test_accuracy(agent, level_path, test_train_split, 0, new_log=True)

        level_count = 0
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')
            train_levels, _ = Trainer.__load_levels(level_path, test_train_split)

            for i, (grid, grid_solved) in enumerate(train_levels, 1):
                agent.environment = environment = OfflineEnvironment(grid, grid_solved)
                Trainer.__run(environment, agent, test_only)

                level_count += 1
                if i % (len(train_levels) // 5) == 0:
                    print(f'>>> {i}/{len(train_levels)}')
                    Trainer.__train_test_accuracy(agent, level_path, test_train_split, level_count)
                    agent.save_model()

    @staticmethod
    def train_online(agent, window, delay=False, test_only=False, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                     discount_rate=DISCOUNT_RATE, exploration_rate=EXPLORATION_RATE, experience_replay=True,
                     double_dqn=False, target_update_interval=1, model_path=None):

        parser = LevelParser(window)
        environment = OnlineEnvironment(parser, delay)

        if not agent:
            agent = Agent(environment, batch_size, learning_rate,
                          discount_rate, exploration_rate, experience_replay,
                          double_dqn, target_update_interval, model_path)
        else:
            agent.environment = environment

        Trainer.__run(environment, agent, test_only)
        agent.save_model()
        return agent

    @staticmethod
    def __run(environment, agent, test_only):
        solved = False
        while not solved:
            current_state = environment.get_state()
            action = agent.get_action(current_state)
            new_state, rewards, solved = environment.step(action)

            if not test_only:
                agent.train((current_state, action, rewards, new_state, solved))


if __name__ == '__main__':
    Trainer.train_offline(7)
