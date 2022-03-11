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
from parse import LevelParser, RESOURCES_PATH

# Reduce the number of messages output by TensorFlow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Default values for model hyperparameters.
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 0.99975
LEARNING_RATE_MIN = 0.00001
DISCOUNT_RATE = 0.05
EXPLORATION_RATE = 0.95
EXPLORATION_RATE_DECAY = 0.99975
EXPLORATION_RATE_MIN = 0.01
EXPERIENCE_REPLAY = False
MAX_REPLAY_MEMORY = 50000
DOUBLE_DQN = False
TARGET_UPDATE_INTERVAL = 1

# Number of levels in the combined training and validation datasets; the rest is for testing.
TRAIN_LEVELS = 1600


class Environment:
    """Defines an environment (i.e., a Hexcells level) in which an agent can act.
       This class should not be instantiated. Rather it acts as the parent of the OnlineEnvironment and OfflineEnvironment classes.
    """

    def __init__(self, grid):
        """Create an environment from a given parsed Hexcells level.

        Args:
            grid (grid.Grid): a parsed Hexcells level.
        """
        self._grid = grid
        self._solver = Solver(None)

        # Define an offset for action indexing based on whether the top-left
        # cell is present or not.
        self._offset = 1 if self._grid[0, 0] is None else 0

        # Define the dimensions of the environment.
        self._state_dims = (grid.rows, grid.cols)
        # Maximum number of cells that can appear.
        self._max_cells = (grid.rows * grid.cols) // 2

        # Translate the initial level state to a matrix form.
        self._state = self._initial_state()

    @property
    def state_dims(self):
        """
        Returns:
            tuple: the dimensions of the environment.
        """
        return self._state_dims

    @property
    def max_cells(self):
        """
        Returns:
            int: the number of possible actions in the environment.
        """
        return self._max_cells

    def get_state(self):
        """
        Returns:
            numpy.ndarray: a matrix representation of the current level state.
        """
        return self._state

    def _initial_state(self):
        """
        Returns:
            numpy.ndarray: a matrix representation of the initial level state.
        """
        # Iterate over cell of the level.
        state = np.zeros(self._state_dims)
        for row in range(self._grid.rows):
            for col in range(self._grid.cols):
                # Set the (i,j)th matrix entry as the representative of the
                # corresponding cell in the level.
                state[row, col] = Environment._cell_to_rep(self._grid[row, col])

        return state

    @staticmethod
    def _cell_to_rep(cell):
        """Represent a cell as a single real value.

        Args:
            cell (grid.Cell): cell to create a representative for.

        Returns:
            float: a real-valued representative for the cell.
        """
        # A blank cell is 0.
        if cell is None:
            return 0.0

        # Black cells range from 0.1 to 0.8
        elif cell.colour == Cell.BLACK:
            if cell.number == '?':
                return 0.8
            else:
                return (cell.number + 1) / 10

        # Blue cells are 0.9
        elif cell.colour == Cell.BLUE:
            return 0.9

        # Orange cells are 1.0
        elif cell.colour == Cell.ORANGE:
            return 1.0

    def _action_to_coords(self, action_index):
        """Convert a 1-dimensional action index to a 2-dimensional row and column index.

        Args:
            action_index (int): index of the action to transform.

        Returns:
            tuple: button action, row and column corresponding to the action index.
        """
        button = action_index // self._max_cells
        action_index %= self.max_cells
        row = (2 * action_index + self._offset) // self._grid.cols
        col = (2 * action_index + self._offset) % self._grid.cols
        return int(button), int(row), int(col)

    def _coords_to_action(self, coords, button):
        """Convert a 2-dimensional (row, column) index to a 1-dimensional action index.

        Args:
            coords (tuple): row and column to transform.
            button (int): either 0 (left click) or 1 (right click).

        Returns:
            int: action index corresponding to the row and column.
        """
        return int((coords[0] * self._grid.cols + coords[1]) // 2) + button * self._max_cells

    def unknown(self):
        """
        Returns:
            list: action indices of the unknown (i.e., orange) cells.
        """
        # Iterate over each unknown cell.
        action_indices = []
        for cell in self._grid.unknown_cells():
            # Record the actions corresponding to left and right clicking each cell.
            action_indices.append(self._coords_to_action(cell.grid_coords, 0))
            action_indices.append(self._coords_to_action(cell.grid_coords, 1))

        return action_indices

    def _rewards(self):
        """Calculates the rewards of all possible actions, given the cells that could be
           clicked with the information currently available.

        Returns:
            tuple: either -1 or +1 for each action, depending on whether the action is "correct" or not.
                   Also returned are the cells that could be left or right clicked.
        """
        # Get the cells that could be left or right clicked, given the available information in the level.
        left_click_cells, right_click_cells = self._solver.solve_single_step(self._grid)
        left_click_coords = [cell.grid_coords for cell in left_click_cells]
        right_click_coords = [cell.grid_coords for cell in right_click_cells]

        # Get the reward of performing each possible action in the current state.
        # Times two because each cell can be either left or right clicked.
        rewards = []
        for action in range(2 * self._max_cells):
            # Get the button click, row and column corresponding to the chosen action index.
            button, row, col = self._action_to_coords(action)

            # Check if the chosen action is correct against the solution.
            if ((row, col) in left_click_coords and button == 0) or ((row, col) in right_click_coords and button == 1):
                rewards.append(1) # Reward is +1 if correct
            else:
                rewards.append(-1) # Reward is -1 if incorrect.

        return rewards, left_click_coords, right_click_coords


class OfflineEnvironment(Environment):
    """Defines an offline environment (i.e., a stored Hexcells level) in which an agent can act."""

    def __init__(self, grid, grid_solved):
        """Initialise the offline environment.

        Args:
            grid (grid.Grid): an unsolved copy of the level.
            grid_solved (grid.Grid): a solved copy of the level.
        """
        # Call the parent constructor and save the solved copy of the level.
        super().__init__(grid)
        self._grid_solved = grid_solved

    def step(self, agent_action):
        """Simulates a single step of an agent acting in the environment.

        Args:
            agent_action (int): index of the agent's selected action

        Returns:
            tuple: level state after performing the action, the reward and whether the level is now solved.
        """
        # Get the reward of performing each action in the current state.
        rewards, _, _ = self._rewards()

        # Update the level state by uncovering the agent's chosen cell.
        # Get the row and column corresponding to the action.
        _, row, col = self._action_to_coords(agent_action)
        cell_curr = self._grid[row, col]
        cell_true = self._grid_solved[row, col]

        # There is no need to update the level state if the action is not uncovering an unknown cell.
        if cell_curr and cell_curr.colour == Cell.ORANGE:
            # If the uncovered cell is blue, decrement the remaining counter.
            if cell_true.colour == Cell.BLUE:
                self._grid.remaining -= 1

            # Update the current cell colour to the true value.
            cell_curr.colour = cell_true.colour
            # Also update the number and hint type if applicable.
            if cell_true.number is not None:
                cell_curr.number = str(cell_true.number)
                cell_curr.hint = cell_true.hint

            # Update the matrix form of the level state using the cell's new representative.
            self._state[row][col] = Environment._cell_to_rep(cell_true)

        # Check if the level is now solved.
        solved = len(self._grid.unknown_cells()) == 0
        return self._state, rewards, solved


class OnlineEnvironment(Environment):
    """Defines an online environment (i.e., a Hexcells level from a live game window) in which an agent can act."""

    def __init__(self, parser, delay):
        """Initialise the online environment.

        Args:
            parser (parse.GameParser): used to parse the level state at each step.
            delay (bool): whether to use a delay after clicking cells for particles to clear.
        """
        # Call the parent constructor with a parsed grid capturing the initial level state.
        super().__init__(parser.parse_grid())
        self._solver =  Solver(parser)
        self.__parser = parser
        self.__delay = delay

    def step(self, agent_action):
        """Simulates a single step of an agent acting in the environment.

        Args:
            agent_action (int): index of the agent's selected action

        Returns:
            tuple: level state after performing the action, the reward and whether the level is now solved.
        """
        # Get the reward of performing each action in the current state.
        rewards, left_click_coords, right_click_coords = self._rewards()

        # Get the button action, row and column corresponding to the agent's action.
        button, row, col = self._action_to_coords(agent_action)
        cell = self._grid[row, col]
        solved = False

        # Check if the cell is in the set of cells that could be left clicked.
        if (row, col) in left_click_coords:
            # If the wrong button action was made, still perform that action to show what the agent chose.
            if rewards[agent_action] == -1:
                # Click the cell but do not try to parse it as the action is incorrect.
                # I.e., the cell will still be unknown.
                self.__parser.click_cells([cell], 'right')
                time.sleep(0.1)

            # If there are at least two unknown cells, left click and parse the chosen cell.
            if len(self._grid.unknown_cells()) > 1:
                _, remaining = self.__parser.parse_clicked(self._grid, [cell], [], self.__delay)
            else:
                # Otherwise, the cell is the last one, so parsing could be problematic due to the level completion screen.
                # Instead, just click the cell and ignore parsing.
                self.__parser.click_cells([cell], 'left')
                cell.colour = Cell.BLUE
                remaining = 0 # Solved as there are no remaining cells to uncover.
                solved = True

        # Check if the cell is in the set of cells that could be right clicked.
        elif (row, col) in right_click_coords:
            # If the wrong button action was made, still perform that action to show what the agent chose.
            if rewards[agent_action] == -1:
                # Click the cell but do not try to parse it as the action is incorrect.
                # I.e., the cell will still be unknown.
                self.__parser.click_cells([cell], 'left')
                time.sleep(0.1)

            # If there are at least two unknown cells, right click and parse the chosen cell.
            if len(self._grid.unknown_cells()) > 1:
                _, remaining = self.__parser.parse_clicked(self._grid, [], [cell], self.__delay)
            else:
                # Otherwise, the cell is the last one, so parsing could be problematic due to the level completion screen.
                # Instead, just click the cell and ignore parsing.
                self.__parser.click_cells([cell], 'right')
                cell.colour = Cell.BLACK
                cell.number = '?'
                remaining = 0  # Solved as there are no remaining cells to uncover.
                solved = True

        else:
            # Otherwise, the chosen cell was not in the cells that could be clicked with the available information.
            # First parse the mistakes made and remaining counter values.
            mistakes_before, remaining = self.__parser.parse_counters()

            # We do not need to worry about the level completion screen as, if there is a single cell left,
            # guessing will not be required and so the solver will identify the cell.
            # I.e., the case will be handled above.

            # Check if the agent chose a left click.
            if button == 0:
                # Left click the cell and then try to parse it.
                mistakes_after, remaining = self.__parser.parse_clicked(self._grid, [cell], [], self.__delay)
                # If a mistake was made, right click the cell and parse it again.
                if mistakes_after > mistakes_before:
                    time.sleep(0.1)
                    _, remaining = self.__parser.parse_clicked(self._grid, [], [cell], self.__delay)

            # Check if the agent chose a right click.
            elif button == 1:
                # Right click the cell and then try to parse it.
                mistakes_after, remaining = self.__parser.parse_clicked(self._grid, [], [cell], self.__delay)
                # If a mistake was made, left click the cell and parse it again.
                if mistakes_after > mistakes_before:
                    time.sleep(0.1)
                    _, remaining = self.__parser.parse_clicked(self._grid, [cell], [], self.__delay)

        # Update the grid and corresponding matrix form.
        self._grid.remaining = remaining
        self._state[row][col] = Environment._cell_to_rep(cell)
        return self._state, rewards, solved


class Agent:
    """Define an agent that can select and perform actions in an environment (i.e., a Hexcells level)."""

    def __init__(self, environment, learning_rate, discount_rate, exploration_rate, experience_replay, double_dqn,
                 batch_size=BATCH_SIZE, target_update_interval=TARGET_UPDATE_INTERVAL, model_path=None):
        """ Initialise the agent using the given hyperparameter values.

        Args:
            environment (learn.Environment): the environment for the agent to act in.
            learning_rate (float): controls model weights are adjusted.
            discount_rate (float): controls how the agent values future actions.
            exploration_rate (float): controls how much the agent explores.
            experience_replay (bool): whether to use experience replay or not.
            double_dqn (bool): whether to use double deep Q-learning or not.
            batch_size (int, optional): sample size to use for experience replay.
            target_update_interval (int, optional): update interval used in double deep Q-learning.
            model_path (str, optional): file path to an existing model to load.
        """
        self.__environment = environment
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__discount_rate = discount_rate
        self.__exploration_rate = exploration_rate

        # Initialise the replay memory as empty.
        self.__replay_memory = []
        # The memory size is 1 if experience replay is not being used.
        self.__max_replay_memory = MAX_REPLAY_MEMORY if experience_replay else 1

        # If a model path is given, try to load the model.
        if model_path is not None:
            if os.path.isfile(model_path):
                self.__model = load_model(model_path)
                self.__model_path = model_path
            else:
                raise FileNotFoundError
        else:
            # Otherwise, create a new model using the dimensions, and maximum number of cells, in the environment.
            self.__model = self.__create_model()

            # Try to get a file name for the new model that does not already exist.
            i = 1
            while True:
                filename = f'model_{i}.h5'
                model_path = os.path.join(RESOURCES_PATH, 'models', filename)
                if not os.path.isfile(model_path):
                    break
                i += 1

            # Save the newly created model.
            self.__model_path = model_path
            self.save_model()


        if double_dqn:
            # If using double deep Q-learning, create a new model with the same architecture as above.
            self.__target_model = self.__create_model()
            # Set the weights to be the same as the main model.
            self.__target_model.set_weights(self.__model.get_weights())
            self.__target_update_interval = target_update_interval
            self.__target_update_counter = 0
        else:
            # Otherwise, ignore the target model.
            self.__target_model = None

    @property
    def environment(self):
        """
        Returns:
            learn.Environment: the environment in which the agent acts.
        """
        return self.__environment

    @environment.setter
    def environment(self, environment):
        """Set the environment in which the agent acts.

        Args:
            environment (learn.Environment): the environment for the agent to act in.
        """
        self.__environment = environment

    @property
    def model_path(self):
        """
        Returns:
            str: file path of the save model.
        """
        return self.__model_path

    def __create_model(self):
        """Creates the convolutional neural network model used by the agent to evaluate actions.

        Returns:
            keras.Sequential: agent's model of the action-value function.
        """
        # Define the input and output dimensions based on the agent's environment.
        input_dims = (*self.__environment.state_dims, 1)
        num_actions = self.__environment.max_cells * 2

        # Create a sequential model consisting of 4 convolutional layers and 4 dense (fully-connected) layers.
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

        # Compile the model using the Adam optimiser and the mean-squared error loss function.
        model.compile(optimizer=Adam(learning_rate=self.__learning_rate), loss='mse')
        return model

    def get_action(self, state, training=True):
        """Get the action chosen by the agent in a given state.

        Args:
            state (numpy.ndarray): the matrix representation of the current level state.
            training (bool, optional): whether to use the epsilon-greedy strategy or not (only used in training).

        Returns:
            int: the index of the agent's chosen action.
        """
        # Get the action indices of the unknown cells in the environment.
        unknown = self.__environment.unknown()

        # Select a randomly chosen action with probability given by the exploration rate.
        if training and np.random.random() < self.__exploration_rate:
            return np.random.choice(self.__environment.unknown())
        else:
            # If a random action was not chosen, predict the expected future reward of each action in the given state.
            action_rewards = self.__model.predict(np.reshape(state, (1, *self.__environment.state_dims, 1)))[0]
            # Set a reward of -infinity for known cells.
            filtered_rewards = np.zeros_like(action_rewards) - float('inf')
            filtered_rewards[unknown] = action_rewards[unknown]
            # Select the action with maximum future reward.
            return np.argmax(filtered_rewards)

    def train(self, transition):
        """Train the agent using a given experience.

        Args:
            transition (tuple): a prior experience to learn from.
        """
        # If the replay memory is full, remove an entry at random from it.
        if len(self.__replay_memory) >= self.__max_replay_memory:
            self.__replay_memory.pop(np.random.randint(0, len(self.__replay_memory)))

        # Add the new experience to the memory.
        self.__replay_memory.append(transition)

        # Get a random batch from the replay memory with size given by the batch size.
        batch = random.sample(self.__replay_memory, min(self.__batch_size, len(self.__replay_memory)))

        # Get the current states and rewards from the batch.
        current_states = np.array([np.expand_dims(s, -1) for s, _, _, _, _ in batch])
        current_state_rewards = np.array([r for _, _, r, _, _ in batch])

        # If the discount rate is non-zero, we need to consider the future reward in each state.
        if self.__discount_rate > 0:
            # Use the target model is using double deep Q-learning.
            predictor = self.__target_model if self.__target_model else self.__model

            # Predict the expected future reward from each of the resulting states in each experience.
            next_states = np.array([np.expand_dims(s, -1) for _, _, _, s, _ in batch])
            next_state_rewards = predictor.predict(next_states)

            # If the level is not solved in each of these resulting states, add the predicted future reward
            # with an adjustment from the discount rate.
            # If the level is solved in the resulting state, there will be no additional rewards.
            for i, (_, action, _, _, solved) in enumerate(batch):
                if not solved:
                    current_state_rewards[i][action] += self.__discount_rate * np.max(next_state_rewards[i])

        # Use the actual rewards in each state to update the agent's model of the action-value function.
        self.__model.fit(current_states, np.array(current_state_rewards), shuffle=False, verbose=False)

        # If using double deep Q-learning, update the target model's weights based on the update interval.
        if self.__target_model:
            # If the level is solved, increment the counter.
            if transition[4]:
                self.__target_update_counter += 1

            # If the desired number of levels has been solved, update the weights of the target model.
            if self.__target_update_counter > self.__target_update_interval:
                self.__target_model.set_weights(self.__model.get_weights())
                self.__target_update_counter = 0 # Reset the counter.

        # Decay the learning and exploration rates.
        self.__learning_rate = max(LEARNING_RATE_MIN, self.__learning_rate * LEARNING_RATE_DECAY)
        self.__exploration_rate = max(EXPLORATION_RATE_MIN, self.__exploration_rate * EXPLORATION_RATE_DECAY)

    def save_model(self):
        """Save the model."""
        self.__model.save(self.__model_path)


class Trainer:
    """Contains the code related to training and testing models."""

    @staticmethod
    def __train_val_accuracy(agent, levels_path, train_val_split, level_count, new_log=False):
        """Compute the training and validation accuracy of an agent on a given set of levels.

        Args:
            agent (learn.Agent): agent to benchmark the performance of.
            levels_path (str): path to the file to load levels from.
            train_val_split (float): proportion of the levels to use for training.
            level_count (int): how many levels have been solved so far during training.
            new_log (bool, optional): whether to create a new log for the accuracies.
        """
        # Get the training and validation datasets.
        print('Computing accuracy...')
        train_levels, val_levels = Trainer.__load_levels(levels_path, train_val_split)

        # Compute the accuracy of the agent in each environment.
        train_accuracy = Trainer.__accuracy(agent, train_levels)
        val_accuracy = Trainer.__accuracy(agent, val_levels)

        # Log the accuracies to the terminal.
        print(f'Training Accuracy: {train_accuracy}')
        print(f'Validation Accuracy: {val_accuracy}\n')

        # Save the accuracies to a file in a directory called "logs".
        save_path, filename = os.path.split(agent.model_path)
        save_path = os.path.join(save_path, 'logs')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Append to the existing log if present.
        filename = f'{os.path.splitext(filename)[0]}.csv'
        file_path = os.path.join(save_path, filename)
        write_mode = 'w' if new_log or not os.path.isfile(file_path) else 'a'

        with open(file_path, write_mode) as file:
            file.write(f'{level_count}, {train_accuracy}, {val_accuracy}\n')

    @staticmethod
    def test_accuracy(model_path, level_size):
        """ Compute the test accuracy of an agent on a given set of levels.

        Args:
            model_path (str): file path of the model to test.
            level_size (str): small, medium or large.
        """
        # Load the levels from the given file path.
        levels_path = os.path.join(RESOURCES_PATH, 'levels', f'levels_{level_size}.pickle')
        with open(levels_path, 'rb') as file:
            levels, _ = pickle.load(file)
            # Skip over the levels in the training and validation datasets.
            test_levels = levels[TRAIN_LEVELS:]

        # Load the model from the given file path.
        environment = OfflineEnvironment(*levels[0]) # Use the first level to establish the environment size.
        agent = Agent(environment, LEARNING_RATE, DISCOUNT_RATE, 0, False, False, model_path=model_path)

        # Compute the agent's accuracy on the test dataset.
        test_accuracy = Trainer.__accuracy(agent, test_levels)
        print(f'Test Accuracy: {test_accuracy}')

    @staticmethod
    def __load_levels(levels_path, train_val_split):
        """Load training and validation datasets from a given file path.

        Args:
            levels_path (str): path to the file containing the levels to load.
            train_val_split (float): proportion of the levels to use for training.

        Returns:
            list: levels loaded from the chosen file.
        """
        # Load the levels from the file
        with open(levels_path, 'rb') as file:
            levels, _ = pickle.load(file)
            levels = levels[:TRAIN_LEVELS] # Only keep the first 1600.

        # Split the levels into training and validation datasets.
        split = round(len(levels) * train_val_split)
        return levels[:split], levels[split:]

    @staticmethod
    def __accuracy(agent, levels):
        """Computes the accuracy of an agent on a given set of levels.

        Args:
            agent (learn.Agent): agent to benchmark the performance of.
            levels (list): levels to benchmark the agent on.

        Returns:
            float: accuracy of the agent (i.e., the probability of the agent selecting a correct action).
        """
        # Check that there is at least one level to calculate the accuracy on.
        if len(levels) == 0:
            raise RuntimeError('Calculating accuracy with no levels')

        # Iterate over each level in the given dataset.
        actions, mistakes = 0, 0
        for i, (grid, grid_solved) in enumerate(levels, 1):
            # Create an environment for the level.
            agent.environment = environment = OfflineEnvironment(grid, grid_solved)

            solved = False
            while not solved:
                # Get the current level state.
                current_state = environment.get_state()

                # Get the agent's chosen action in the state.
                # Ignores the epsilon-greedy strategy (only used in training).
                action = agent.get_action(current_state, training=False)

                # Perform the agent's chosen action.
                _, rewards, solved = environment.step(action)

                # Log a mistake if one was made.
                actions += 1
                if rewards[action] != 1:
                    mistakes += 1

            # Display progress every 100 levels.
            if i % 100 == 0:
                print(f'>>> {i}/{len(levels)}')

        # Compute the accuracy over all levels.
        return 1 - (mistakes / actions)

    @staticmethod
    def run_offline(epochs, training, learning_rate=LEARNING_RATE, discount_rate=DISCOUNT_RATE,
                      exploration_rate=EXPLORATION_RATE, experience_replay=EXPERIENCE_REPLAY, double_dqn=DOUBLE_DQN,
                      batch_size=BATCH_SIZE, model_path=None, level_size='small', train_val_split=0.8125):
        """Run an agent in a number of offline learning environments.

        Args:
            epochs (int): number of times to train the agent over the entire dataset.
            training (bool, optional): whether to train the agent or not.
            learning_rate (float, optional): controls model weights are adjusted.
            discount_rate (float, optional): controls how the agent values future actions.
            exploration_rate (float, optional): controls how much the agent explores.
            experience_replay (bool, optional): whether to use experience replay or not.
            double_dqn (bool, optional): whether to use double deep Q-learning or not.
            batch_size (int, optional): sample size to use for experience replay.
            model_path (str, optional): file path to an existing model to load.
            level_size (str, optional): level size to train on.
            train_val_split (float, optional): proportion of levels to use for training.
        """
        # Load a single level to establish the dimensions of the environment.
        level_path = os.path.join(RESOURCES_PATH, 'levels', f'levels_{level_size}.pickle')
        train_levels, _ = Trainer.__load_levels(level_path, 1)
        environment = OfflineEnvironment(*train_levels[0])

        # Create an agent using the given hyperparameter values.
        agent = Agent(environment, learning_rate, discount_rate, exploration_rate,
                      experience_replay, double_dqn, batch_size, model_path=model_path)

        # Compute the initial accuracy on the training and validation datasets. Create a new log.
        Trainer.__train_val_accuracy(agent, level_path, train_val_split, level_count=0, new_log=True)

        # Iterate over the training dataset for the given number of epochs.
        level_count = 0
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')
            # Load the levels on each epoch to reset their state.
            train_levels, _ = Trainer.__load_levels(level_path, train_val_split)

            # Iterate over each level in the training dataset.
            for i, (grid, grid_solved) in enumerate(train_levels, 1):
                # Set the agent's environment to the level and run it.
                agent.environment = environment = OfflineEnvironment(grid, grid_solved)
                Trainer.__run(agent, environment, training)

                # Compute and log the training and validation accuracy at 5 points during training.
                level_count += 1
                if i % (len(train_levels) // 5) == 0:
                    print(f'>>> {i}/{len(train_levels)}')
                    Trainer.__train_val_accuracy(agent, level_path, train_val_split, level_count)

                    # Only save the model if training was specified.
                    if training:
                        agent.save_model()

    @staticmethod
    def run_online(agent, window, delay=False, training=False, learning_rate=LEARNING_RATE,
                   discount_rate=DISCOUNT_RATE, exploration_rate=EXPLORATION_RATE, experience_replay=EXPERIENCE_REPLAY,
                   double_dqn=DOUBLE_DQN, batch_size=BATCH_SIZE, model_path=None):
        """Runs an agent in an online learning environment.

        Args:
            agent (learn.Agent): agent to run in an online environment.
            window (window.Window): the active game window to run the agent on.
            delay (bool, optional): whether to use a delay after clicking cells.
            training (bool, optional): whether train the agent or not.
            learning_rate (float, optional): controls model weights are adjusted.
            discount_rate (float, optional): controls how the agent values future actions.
            exploration_rate (float, optional): controls how much the agent explores.
            experience_replay (bool, optional): whether to use experience replay or not.
            double_dqn (bool, optional): whether to use double deep Q-learning or not.
            batch_size (int, optional): sample size to use for experience replay.
            model_path (str, optional): file path to an existing model to load.

        Returns:
            learn.Agent: the after being run on the online environment.
        """
        # Create an online environment for the active game window.
        parser = LevelParser(window)
        environment = OnlineEnvironment(parser, delay)

        # If an agent was not given, create one.
        if not agent:
            agent = Agent(environment, learning_rate, discount_rate, exploration_rate,
                          experience_replay, double_dqn, batch_size, TARGET_UPDATE_INTERVAL, model_path)
        else:
            # Otherwise, update the environment of the given agent.
            agent.environment = environment

        # Run the agent on the online environment.
        Trainer.__run(agent, environment, training)

        # If training, save the model.
        if training:
            agent.save_model()

        return agent

    @staticmethod
    def __run(agent, environment, training):
        """Run an agent on a given environment.

        Args:
            agent (learn.Agent): agent to run in the environment.
            environment (learn.Environment): environment for the agent to act in
            training (bool): whether the agent learns from experiences.
        """
        solved = False
        while not solved:
            # Get the current state of the environment.
            current_state = environment.get_state()
            # Get the action chosen by the agent.
            action = agent.get_action(current_state, training)
            # Perform the agent's chosen action.
            new_state, rewards, solved = environment.step(action)

            # If training, use the experience to update the agent's model.
            if training:
                agent.train((current_state, action, rewards, new_state, solved))


if __name__ == '__main__':
    # Train a new model offline for 7 epochs.
    Trainer.run_offline(7, training=True)

    # Test the accuracy of the model architecture defined above on three different level sizes.
    # Note that these models are pre-trained.
    Trainer.test_accuracy('resources/models/model_4.h5', 'small')
    Trainer.test_accuracy('resources/models/model_12.h5', 'medium')
    Trainer.test_accuracy('resources/models/model_13.h5', 'large')
