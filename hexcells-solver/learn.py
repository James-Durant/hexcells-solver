import os
import pickle5 as pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

from grid import Cell
from parse import GameParser
from window import get_window

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Environment:
    STATE_DIMS = (7, 3)
    MAX_HEXES = 10

    def get_state(self):
        return self._state

    def _initial_state(self):
        state = np.zeros(Environment.STATE_DIMS)
        self._row_offset = 1 if self._grid[0, 1] is None else 0

        for row in range(self._grid.rows):
            for col in range(self._grid.cols):
                state[row + self._row_offset, col] = self._cell_to_rep(self._grid[row, col])

        return state

    @staticmethod
    def _cell_to_rep(cell):
        if cell is None:
            return 0

        elif cell.colour == Cell.BLACK:
            if cell.digit == '?':
                return 8
            else:
                return cell.digit + 1

        elif cell.colour == Cell.BLUE:
            return 9

        elif cell.colour == Cell.ORANGE:
            return 10

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
        self._grid = deepcopy(grid)
        self._grid_solved = grid_solved
        self._state = self._initial_state()

    def step(self, action_index):
        rewards = []
        for action in range(2 * Environment.MAX_HEXES):
            state, grid, reward, solved = self.__step(action_index)
            rewards.append(reward)
            if action == action_index:
                grid_new = grid
                state_new = state
                solved_new = solved

        self._grid = grid_new
        self._state = state_new

        return state_new, rewards, solved_new

    def __step(self, action_index):
        state = deepcopy(self._state)
        grid = deepcopy(self._grid)

        button = action_index // Environment.MAX_HEXES  # Either left or right click
        action_index %= Environment.MAX_HEXES
        row, col = self._action_to_coords(action_index)

        reward = -1
        solved = False
        if row in range(Environment.STATE_DIMS[0]) and col in range(Environment.STATE_DIMS[1]):
            cell_curr = grid[row - self._row_offset, col]
            cell_true = self._grid_solved[row - self._row_offset, col]

            if cell_curr and cell_curr.colour == Cell.ORANGE:
                if ((button == 0 and cell_true.colour == Cell.BLUE) or
                        (button == 1 and cell_true.colour == Cell.BLACK)):
                    reward = 1

                if cell_true.colour == Cell.BLUE:
                    grid.remaining -= 1

                cell_curr.colour = cell_true.colour
                if cell_true.digit is not None:
                    cell_curr.hint = cell_true.hint
                    cell_curr.digit = str(cell_true.digit)

                state[row][col] = self._cell_to_rep(cell_true)

                if len(grid.unknown_cells()) == 0:
                    solved = True

        return state, grid, reward, solved


class OnlineEnvironment(Environment):
    def __init__(self, parser):
        self.__parser = parser
        self._grid = self.__parser.parse_grid()
        self._state = self._initial_state()

    def step(self, action_index):
        button = action_index // Environment.MAX_HEXES  # Either left or right click
        action_index %= Environment.MAX_HEXES
        row, col = self._action_to_coords(action_index)

        reward = -2
        solved = False
        if row in range(Environment.STATE_DIMS[0]) and col in range(Environment.STATE_DIMS[1]):
            cell = self._grid[row - self._row_offset, col]
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


class Agent:
    def __init__(self, environment=None, epsilon=0.95, discount=0.1,
                 learning_rate=0.01, batch_size=64,
                 max_replay_memory=50000,
                 weights_path=r'./resources/models/model_weights.h5'):

        self.__environment = environment
        self.__epsilon = epsilon
        self.__discount = discount
        self.__learning_rate = learning_rate
        self.__batch_size = batch_size
        self.__max_replay_memory = max_replay_memory
        self.__replay_memory = []
        self.__weights_path = weights_path

        self.__model = self.__create_model((*Environment.STATE_DIMS, 1), Environment.MAX_HEXES * 2)

        if os.path.isfile(self.__weights_path):
            self.__model.load_weights(self.__weights_path)

    @property
    def environment(self):
        return self.__environment

    @environment.setter
    def environment(self, environment):
        self.__environment = environment

    def __create_model(self, input_dims, num_actions):
        model = Sequential([Conv2D(21, (3, 3), activation='relu', padding='same', input_shape=input_dims),
                            Conv2D(21, (3, 3), activation='relu', padding='same'),
                            Conv2D(21, (3, 3), activation='relu', padding='same'),
                            Conv2D(21, (3, 3), activation='relu', padding='same'),
                            Flatten(),
                            Dense(256, activation='relu'),
                            Dense(256, activation='relu'),
                            Dense(num_actions, activation='linear')])

        model.compile(optimizer=Adam(learning_rate=self.__learning_rate, epsilon=1e-4), loss='mse')
        return model

    def get_action(self, state):
        unknown = self.__environment.unknown()

        if np.random.random() < self.__epsilon:
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

        if len(self.__replay_memory) < self.__batch_size:
            return

        np.random.shuffle(self.__replay_memory)

        current_states = np.array([s for s, _, _, _, _ in self.__replay_memory])
        rewards = np.array([r for _, _, r, _, _ in self.__replay_memory])

        # new_states = np.array([t[3] for t in self.__replay_memory])
        # new_state_rewards = self.__model.predict(np.expand_dims(new_states, -1))

        # y = []
        # for i, (current_state, action, rewards, new_current_state, solved) in enumerate(self.__replay_memory):
        #    current_state_rewards[i] = rewards
        #    if not solved:
        #        current_state_rewards[i][action] += self.__discount*np.max(new_state_rewards[i])

        #    y.append(current_state_rewards[i])

        self.__model.fit(current_states, np.array(rewards), batch_size=self.__batch_size,
                         shuffle=False, verbose=False)

    def save_weights(self):
        self.__model.save_weights(self.__weights_path)


# online training?
def train_offline(epochs=50, levels_path='resources/levels/levels.pickle'):
    with open(levels_path, 'rb') as file:
        levels, _ = pickle.load(file)
        train_levels = levels[:10]
        test_levels = levels[10:]

    agent = Agent()
    train_accuracy = accuracy_offline(agent, train_levels)
    test_accuracy = accuracy_offline(agent, test_levels)
    print('Training Accuracy: {0}'.format(train_accuracy))
    print('Test Accuracy: {0}\n'.format(test_accuracy))

    for epoch in range(epochs):
        print('Epoch {}'.format(epoch + 1))

        for i, (grid, grid_solved) in enumerate(levels, 1):
            agent.environment = environment = OfflineEnvironment(grid, grid_solved)

            solved = False
            while not solved:
                current_state = environment.get_state()
                action = agent.get_action(current_state)
                new_state, reward, solved = environment.step(action)
                agent.train((current_state, action, reward, new_state, solved))

            if i % 100 == 0:
                print('>>> {0}/{1}'.format(i, len(train_levels)))
                agent.save_weights()

        agent.save_weights()
        train_accuracy = accuracy_offline(agent, train_levels)
        test_accuracy = accuracy_offline(agent, test_levels)
        print('Training Accuracy: {0}'.format(train_accuracy))
        print('Test Accuracy: {0}\n'.format(test_accuracy))


def accuracy_offline(agent, levels):
    actions = 0
    mistakes = 0
    for grid, grid_solved in levels:
        agent.environment = environment = OfflineEnvironment(grid, grid_solved)

        solved = False
        while not solved:
            current_state = environment.get_state()
            action = agent.get_action(current_state, testing=True)
            _, reward, solved = environment.step(action)

            actions += 1
            if reward != 1:
                mistakes += 1

    return 1 - (mistakes / actions)


# def test_online():
#    environment = OnlineEnvironment(GameParser(get_window()))
#    agent = Agent(environment, epsilon=0)
#
#    solved = False
#    while not solved:
#        current_state = environment.get_state()
#        action = agent.get_action(current_state)
#        new_state, reward, solved = environment.step(action)

if __name__ == '__main__':
    train_offline()
    # test_offline(Agent())
    # test_online()
