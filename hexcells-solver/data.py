import os
import cv2
import json
import time
import pickle
import sys
import pyperclip

from subprocess import Popen

from grid import Cell
from solve import Solver
from navigate import Navigator
from parse import average_hash, LevelParser, MenuParser, RESOLUTIONS, STEAM_PATH, RESOURCES_PATH

# Steam IDs of the three Hexcells games.
GAMEIDS = {'Hexcells': '265890', 'Hexcells Plus': '271900', 'Hexcells Infinite': '304410'}


class Generator:
    """The parent class for the LearningData and ImageData classes containing code common to both."""

    def __init__(self):
        """Store the paths to the Hexcells options.txt file and Steam executable."""
        self.__options_path = os.path.join(STEAM_PATH, 'steamapps\common\{0}\saves\options.txt')
        self.__steam_path = os.path.join(STEAM_PATH, 'steam.exe')

    def _load_game(self, game, resolution, use_level_hashes=True, use_screen_hashes=True):
        """Load a given game at a resolution, closing the game currently current if necessary.

        Args:
            game (str): game to load.
            resolution (tuple): screen width and screen height values defining game resolution.
            use_level_hashes (bool, optional): whether to load level hashes.
            use_screen_hashes (bool, optional): whether to load screen hashes.

        Returns:
            navigate.Navigator: the newly opened game window encapsulated as a Navigator object.
        """
        # Try to access a running game window and close it.
        try:
            menu = Navigator(use_level_hashes, use_screen_hashes)
            menu.close_game()

        except KeyboardInterrupt:
            # If the program receives Control + C, exit.
            sys.exit()

        # Ignore all other exceptions and continue (a game window was not open).
        except Exception:
            pass

        # Open the JSON options file for the selected game.
        options_path = self.__options_path.format(game)
        with open(options_path, 'r') as file:
            data = json.load(file)

        # Write the chosen resolution to the options file.
        data['screenWidth'], data['screenHeight'] = resolution
        with open(options_path, 'w') as file:
            json.dump(data, file)

        # Run the chosen game by sending the appropriate command to the Steam executable.
        Popen([self.__steam_path, f'steam://rungameid/{GAMEIDS[game]}'],
              shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

        # Wait until the game window has loaded.
        time.sleep(7)

        # Try to capture the newly opened game window.
        menu = Navigator(use_level_hashes, use_screen_hashes)
        menu.window.move_mouse()
        menu.wait_until_loaded() # Make sure the main menu screen has fully loaded.
        return menu


class LearningData(Generator):
    """Contains the code for generating training data (i.e., levels and their pre-computed solutions) for deep Q-learning."""

    def __init__(self):
        """Cell the parent constructor."""
        super().__init__()

    def make_dataset(self, num_levels):
        """Creates a dataset consisting of levels and associated solution.

        Args:
            num_levels (int): number of levels to generate.
        """
        # Load Hexcells Infinite at a resolution of 1920x1080.
        menu = self._load_game('Hexcells Infinite', (1920, 1080))

        # Parse the buttons on the main menu screen.
        menu_parser = MenuParser(menu.window)
        buttons = menu_parser.parse_main_menu()

        # Click on the level generator button and wait for it to load.
        generator_button = buttons['level_generator']
        menu.window.click(generator_button)
        menu.wait_until_loaded()

        # Create a list of seeds for the random level generator.
        # Each level corresponds to a seed consisting of 8 digits.
        seeds = ['0' * (8 - len(str(i))) + str(i) for i in range(1, num_levels+1)]

        # Keep iterating until there are no levels left to process.
        i = 0
        while i < num_levels:
            # Copy the level's seed to the clipboard.
            pyperclip.copy(seeds[i])

            # Parse the buttons on the level generation screen.
            buttons = menu_parser.parse_generator()

            # Click on the seed button to input the level's seed.
            play_button, seed_button = buttons['play'], buttons['seed']
            menu.window.click(seed_button)
            time.sleep(0.2)
            menu.window.paste() # Paste the level's seed.
            time.sleep(0.2)

            # Click the play button to generate the level for the chosen seed.
            # Click multiple times as there were some issues with only once.
            menu.window.click(play_button, move_mouse=False)
            time.sleep(0.1)
            menu.window.click(play_button, move_mouse=False)
            menu.window.click(play_button, move_mouse=True)
            menu_parser.wait_until_loaded()

            # Parse the initial level state.
            game_parser = LevelParser(menu.window)
            grid = game_parser.parse_grid() # The grid that will be solved to get ground truth solution.
            grid_initial = game_parser.parse_grid() # The grid that is stored as the initial level state.

            # Keep iterating until the level is solved.
            solver = Solver(game_parser)
            while True:
                # Get the cells that can be uncovered, given the current information.
                left_click_cells, right_click_cells = solver.solve_single_step(grid, menu.window, None)

                # Check if there are cells in the level still to uncover.
                if len(left_click_cells) + len(right_click_cells) - len(grid.unknown_cells()) != 0:
                    # If so, click and parse the cells like normal.
                    _, remaining = game_parser.parse_clicked(grid, left_click_cells, right_click_cells)
                    grid.remaining = remaining

                else:
                    # Otherwise, we need to handle the last step carefully.
                    skip = False

                    # If there is only 1 cell to click, skip this level as there will be no guarantee that
                    # its contents can be parsed correctly (due to the level completion screen obscuring it).
                    if len(left_click_cells) + len(right_click_cells) == 1:
                        skip = True

                    # If there is at least one cell to left click, click and parse all identified cells,
                    # leave one left click cell.
                    elif len(left_click_cells) > 0:
                        game_parser.parse_clicked(grid, left_click_cells[1:], right_click_cells)
                        left_click_cells, right_click_cells = [left_click_cells[0]], []

                    # If there is at least one cell to right click, click and parse all identified cells,
                    # but leave one right click cell.
                    elif len(right_click_cells) > 0:
                        game_parser.parse_clicked(grid, left_click_cells, right_click_cells[1:])
                        left_click_cells, right_click_cells = [], [right_click_cells[0]]

                    # Click, but do not parse, the single cell left.
                    game_parser.click_cells(left_click_cells, 'left')
                    game_parser.click_cells(right_click_cells, 'right')

                    # Now that the level has been solved, parse the level completion screen.
                    menu.window.move_mouse()
                    time.sleep(1.2)
                    _, menu_button = menu_parser.parse_level_completion()

                    # Click on the button to return to the level generator screen.
                    menu.window.click(menu_button)
                    menu.wait_until_loaded()

                    # Skip the level if identified above.
                    if skip:
                        print(f'>>> Skipping {i + 1}/{num_levels}')
                        break

                    # Parse the level generator screen buttons.
                    buttons = menu_parser.parse_generator()
                    play_button = buttons['play']

                    # Click the play button to generate the same level again.
                    menu.window.click(play_button, move_mouse=False)
                    time.sleep(0.1)
                    menu.window.click(play_button, move_mouse=False)
                    menu.window.click(play_button, move_mouse=True)
                    menu_parser.wait_until_loaded()

                    # Now click and parse the single cell left from before.
                    game_parser.parse_clicked(grid, left_click_cells, right_click_cells)

                    # Exit the level by pressing the escape key and parsing the level exit screen.
                    menu.window.press_key('esc')
                    menu.window.move_mouse()
                    time.sleep(1.5)
                    _, _, exit_button = menu_parser.parse_level_exit()
                    menu.window.click(exit_button)
                    menu.wait_until_loaded()

                    # If the file to save levels already exists, load the contents.
                    file_path = os.path.join(RESOURCES_PATH, 'levels', 'levels.pickle')
                    if os.path.isfile(file_path):
                        with open(file_path, 'rb') as file:
                            levels, labels = pickle.load(file)
                    else:
                        levels, labels = [], []

                    # Add the newly generated and solved level to the existing levels (if there are any).
                    levels.append((grid_initial, grid))
                    labels.append(seeds[i])

                    # Write the new dataset back to the save file.
                    with open(file_path, 'wb') as file:
                        pickle.dump((levels, labels), file, protocol=pickle.HIGHEST_PROTOCOL)

                    # Move on to the next level to generate.
                    print(f'>>> {i + 1}/{num_levels}')
                    i += 1
                    break


class ImageData(Generator):
    """Contains the code for generating the hashes used in the level and menu parsing algorithms."""

    def __init__(self):
        """Call the parent constructor."""
        super().__init__()

    def make_dataset(self, dataset_type):
        """Creates a dataset of image hashes and associated labels of a chosen type.

        Args:
            dataset_type (str): dataset type to generate.
        """
        # Call a separate method if dealing with screen hashes.
        if dataset_type == 'screen':
            self.__make_dataset_screen()
            return

        # Create the file path to the directory to save the hashes to.
        if dataset_type == 'blue_special':
            hash_path = os.path.join(RESOURCES_PATH, 'blue', 'hashes.pickle')
            game = 'Hexcells Plus'
        else:
            # The game is Hexcells Plus for the "special" blue hashes and Hexcells Infinite otherwise.
            hash_path = os.path.join(RESOURCES_PATH, dataset_type, 'hashes.pickle')
            game = 'Hexcells Infinite'

        # Iterate over each resolution to generate hashes for.
        hashes, labels = [], []
        for resolution in RESOLUTIONS:
            # Load the appropriate game at the resolution.
            # Only load hashes if not generating the level selection dataset.
            use_level_hashes = dataset_type != 'level_select'
            menu = self._load_game(game, resolution, use_level_hashes, False)

            # The column dataset is split into normal, consecutive and non-consecutive hint types.
            if dataset_type == 'column':
                hashes_res, labels_res = [], []
                # For each of the separate hint types.
                for hint in ['normal', 'consecutive', 'non-consecutive']:
                    # Load the custom level specific to the hint type.
                    level_path = os.path.join(RESOURCES_PATH, dataset_type, f'{hint}_level.hexcells')
                    menu.load_custom_level(level_path, screen='main_menu')

                    # Get the hashes from the custom level.
                    hashes_hint, labels_hint = self.__get_hashes(menu.window, f'{dataset_type}_{hint}')
                    # Append them to the rest of the column dataset.
                    hashes_res += hashes_hint
                    labels_res += labels_hint

                    # Exit the level to load the next hint type.
                    menu.exit_level(screen='in_level')

            # The diagonal dataset is split into two parts for each hint type.
            elif dataset_type == 'diagonal':
                hashes_res, labels_res = [], []
                # Iterate over both halves for each hint type.
                for part in ['1', '2']:
                    for hint in ['normal', 'consecutive', 'non-consecutive']:
                        # Load the custom level for the hint type and part.
                        level_path = os.path.join(RESOURCES_PATH, dataset_type, f'{hint}_{part}_level.hexcells')
                        menu.load_custom_level(level_path, screen='main_menu')

                        # Get the hashes from the custom level.
                        hashes_hint, labels_hint = self.__get_hashes(menu.window, f'{dataset_type}_{hint}_{part}')
                        # Append them to the rest of the diagonal dataset.
                        hashes_res += hashes_hint
                        labels_res += labels_hint

                        # Exit the level to load the next hint type.
                        menu.exit_level(screen='in_level')

            else:
                if dataset_type == 'level_select':
                    # Load the first save slot to transition to the level select screen.
                    menu.load_save(1, screen='main_menu')

                elif dataset_type == 'blue_special':
                    # Load the first save slot to transition to the level select screen.
                    menu.load_save(1, screen='main_menu')

                    # Parse the level select screen and click on level 4-6.
                    menu_parser = MenuParser(menu.window, use_level_hashes=True, use_screen_hashes=False)
                    levels = menu_parser.parse_level_selection()
                    menu.window.click(levels['4-6'])
                    menu.wait_until_loaded()

                else:
                    # In all other cases, load the singular custom level for the dataset type.
                    level_path = os.path.join(RESOURCES_PATH, dataset_type, 'level.hexcells')
                    menu.load_custom_level(level_path, screen='main_menu')

                # Get the hashes for the level.
                hashes_res, labels_res = self.__get_hashes(menu.window, dataset_type)

            # Add the hashes for this resolution to the overall dataset.
            hashes += hashes_res
            labels += labels_res

            # Close the game to prepare for loading the next resolution.
            menu.close_game()

            # The counter dataset need only be run for one resolution.
            if dataset_type == 'counter':
                break
        
        # For the special blue dataset, add the hashes and labels to the existing blue dataset.
        if dataset_type == 'blue_special':
            with open(hash_path, 'rb') as file:
                blue_hashes, blue_labels = pickle.load(file)
                hashes += blue_hashes
                labels += blue_labels

        # Finally, record the dataset that combines the hashes and labels from all resolutions.
        with open(hash_path, 'wb') as file:
            pickle.dump((hashes, labels), file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def __get_hashes(window, dataset_type):
        """Get the hashes from a game window for a given dataset type.

        Args:
            window (window.Window): window to extract hashes from.
            dataset_type (str): dataset type to generate.

        Returns:
            tuple: extracted hashes and associated labels.
        """
        # In all the following, use_hashes is set as false to prevent
        # loading of hashes, as they do not exist yet!
        if dataset_type == 'level_select':
            parser = MenuParser(window, use_level_hashes=False, use_screen_hashes=False)
            labels = ['3-3', '3-2', '2-3', '2-2', '3-4', '3-1',
                      '2-4', '2-1', '3-5', '3-6', '2-5', '2-6',
                      '4-3', '4-2', '1-3', '1-2', '4-4', '4-1',
                      '1-4', '1-1', '4-5', '4-6', '1-5', '1-6',
                      '5-3', '5-2', '6-3', '6-2', '5-4', '5-1',
                      '6-4', '6-1', '5-5', '5-6', '6-5', '6-6']

            # Parse the level select screen.
            hashes = parser.parse_level_selection(use_hashes=False)

        elif dataset_type in ['black', 'blue', 'blue_special', 'counter']:
            # Take a screenshot of the game window.
            screenshot = window.screenshot()
            parser = LevelParser(window, use_cell_counter_hashes=False, use_grid_hashes=False)

            if dataset_type == 'counter':
                # The custom level for the counter dataset starts out with 476 unknown cells.
                remaining = 476
                labels = list(range(remaining, -1, -1))

                hashes = []
                mistakes_hash = None
                # Iterate over each of the 476 orange cells.
                for cell in parser.parse_cells(screenshot, Cell.ORANGE):
                    # Get hashes of the counter values.
                    mistakes_hash, remaining_hash = parser.parse_counters(screenshot, use_hashes=False)
                    hashes.append(remaining_hash)

                    # Click on the cell to reduce the "remaining blue cell" counter.
                    window.click_cell(cell, 'left')
                    remaining -= 1

                    # Take a screenshot of the new level state.
                    screenshot = window.screenshot()

                # The hash of the "mistakes made" counter is of the value 0.
                hashes.append(mistakes_hash)

            elif dataset_type == 'black':
                # Get the hashes of the black cells.
                hashes = parser.parse_cells(screenshot, Cell.BLACK, use_hashes=False)
                # The ground truth labels for the custom level.
                labels = ['{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '1', '2', '3',
                          '4', '5', '6', '?', '-2-', '-3-', '-4-', '{0}', '0']

            elif dataset_type == 'blue':
                # Get the hashes of the blue cells.
                hashes = parser.parse_cells(screenshot, Cell.BLUE, use_hashes=False)
                labels = [str(i) for i in range(0, 19)]

            elif dataset_type == 'blue_special':
                # Get the hashes of the blue cells.
                hashes = parser.parse_cells(screenshot, Cell.BLUE, use_hashes=False)
                labels = ['{5}', '-2-', '5', '4', '2', '4', '5', '{4}', '{2}', '5', '3', '10', '10']

            else:
                # The datatype given was not valid.
                raise RuntimeError('Invalid dataset type.')

        else:
            # The ground truth labels for each of the column and diagonal custom levels.
            # Take a look at the levels in resources/levels to understand these labels.
            if dataset_type == 'column_normal':
                labels = [str(i) for i in range(0, 17)]

            elif dataset_type == 'column_consecutive':
                labels = ['{' + str(i) + '}' for i in range(0, 17)]

            elif dataset_type == 'column_non-consecutive':
                labels = ['-' + str(i) + '-' for i in range(2, 16)]

            elif dataset_type == 'diagonal_normal_1':
                labels = ([str(i) for i in range(1, 16)] +
                          ['0', '0'] +
                          [str(i) for i in range(15, 0, -1)])

            elif dataset_type == 'diagonal_normal_2':
                labels = ['24', '25', '27', '26', '23', '22', '21', '20',
                          '19', '18', '17', '16', '16', '17', '18', '19',
                          '20', '23', '21', '24', '26', '27', '25', '22']

            elif dataset_type == 'diagonal_consecutive_1':
                labels = (['{' + str(i) + '}' for i in range(1, 16)] +
                          ['{0}', '{0}'] +
                          ['{' + str(i) + '}' for i in range(15, 0, -1)])

            elif dataset_type == 'diagonal_consecutive_2':
                labels = ['{24}', '{25}', '{27}', '{26}', '{23}', '{22}',
                          '{21}', '{20}', '{19}', '{18}', '{17}', '{16}',
                          '{16}', '{17}', '{18}', '{19}', '{20}', '{23}',
                          '{21}', '{24}', '{26}', '{27}', '{25}', '{22}']

            elif dataset_type == 'diagonal_non-consecutive_1':
                labels = (['-' + str(i) + '-' for i in range(2, 16)] +
                          ['-' + str(i) + '-' for i in range(15, 1, -1)])

            elif dataset_type == 'diagonal_non-consecutive_2':
                labels = ['-23-', '-24-', '-27-', '-26-', '-25-', '-22-',
                          '-21-', '-20-', '-19-', '-18-', '-17-', '-16-',
                          '-16-', '-17-', '-18-', '-19-', '-20-', '-21-',
                          '-24-', '-25-', '-26-', '-22-', '-23-']
            else:
                raise RuntimeError('Invalid dataset type.')

            # Get the hashes of the grid constraints.
            parser = LevelParser(window, use_cell_counter_hashes=True, use_grid_hashes=False)
            hashes = parser.parse_grid(use_hashes=False)

        # Make sure that no numbers were missed when parsing.
        assert len(hashes) == len(labels)
        return hashes, labels

    def __make_dataset_screen(self):
        """Generates the hashes of the different menu screens in Hexcells."""
        # Iterate over each resolution to obtain data for.
        for resolution in RESOLUTIONS:
            images, labels = [], []

            # Load Hexcells and take a screenshot of the main menu screen.
            menu = self._load_game('Hexcells', resolution, use_screen_hashes=False)
            main_menu = menu.window.screenshot()
            images.append(main_menu)
            labels.append('main_menu')
            menu.close_game()

            # Load Hexcells Plus and take a screenshot of the main menu screen.
            menu = self._load_game('Hexcells Plus', resolution, use_screen_hashes=False)
            main_menu = menu.window.screenshot()
            images.append(main_menu)
            labels.append('main_menu')
            menu.close_game()

            # Load Hexcells Infinite and take a screenshot of the main menu screen.
            # The reason that Hexcells and Hexcells Plus menu screens are captured is that
            # each menu screen is different.
            menu = self._load_game('Hexcells Infinite', resolution, use_screen_hashes=False)
            main_menu = menu.window.screenshot()
            images.append(main_menu)
            labels.append('main_menu')

            # Parse the main menu screen.
            menu_parser = MenuParser(menu.window, use_screen_hashes=False)
            buttons = menu_parser.parse_main_menu()
            save_slot_buttons = buttons['save_slots']
            # Click on save slot 1
            menu.window.click(save_slot_buttons[0])
            menu_parser.wait_until_loaded()

            # Take a screenshot of the level select screen.
            level_select = menu.window.screenshot()
            images.append(level_select)
            labels.append('level_select')

            # Parse the level select screen and click on level 1-1.
            levels = menu_parser.parse_level_selection()
            coords = levels['1-1']
            menu.window.click(coords)

            # Press the escape key and take a screenshot of the resulting level exit screen.
            time.sleep(2)
            menu.back()
            level_exit = menu.window.screenshot()

            # Go back into the level and solve it.
            menu.back()
            game_parser = LevelParser(menu.window)
            solver = Solver(game_parser)
            solver.solve(menu.window.title, '1-1')

            # Take a screenshot of the resulting level completion screen.
            menu.window.move_mouse()
            time.sleep(1.2)
            level_completion = menu.window.screenshot()
            images.append(level_completion)
            labels.append('level_completion')

            # Parse the level completion screen and click on the button to return to the main menu.
            _, menu_button = menu_parser.parse_level_completion()
            menu.window.click(menu_button)
            menu.wait_until_loaded()

            # Go back to the main menu screen, parse it, and click on the level generator button.
            menu.back()
            buttons = menu_parser.parse_main_menu()
            generator_button = buttons['level_generator']
            menu.window.click(generator_button)
            menu.wait_until_loaded()

            # Take a screenshot of the level generator screen.
            level_generator = menu.window.screenshot()
            images.append(level_generator)
            labels.append('level_generator')

            # Go back to the main menu screen, parse it, and click on the user levels button.
            menu.back()
            pyperclip.copy('') # Empty the clipboard.
            buttons = menu_parser.parse_main_menu()
            user_levels_button = buttons['user_levels']
            menu.window.click(user_levels_button)
            time.sleep(4) # Wait for the levels to show up.

            # Take a screenshot of the user levels screen.
            user_levels = menu.window.screenshot()
            images.append(user_levels)
            labels.append('user_levels')

            # Go back to the main menu screen, parse it, and click on the options button.
            menu.back()
            buttons = menu_parser.parse_main_menu()
            options_button = buttons['options']
            menu.window.click(options_button)
            time.sleep(2)

            # take a screenshot of the options screen.
            options = menu.window.screenshot()
            images.append(options)
            labels.append('options')

            # Resize each of the screenshots from above and calculate their hash.
            hashes = [average_hash(cv2.resize(image, MenuParser.SCREEN_HASH_DIMS, interpolation=cv2.INTER_AREA))
                      for image in images]

            # Handle the level exit screen separately. Instead of hashing, just thesholded it and resize.
            level_exit = cv2.inRange(level_exit, (180, 180, 180), (255, 255, 255))
            hashes.append(cv2.resize(level_exit, MenuParser.SCREEN_HASH_DIMS))
            labels.append('level_exit')

            # If there is not already a directory for this resolution, create one.
            hash_dir = os.path.join(RESOURCES_PATH, 'screen', '{0}x{1}'.format(*resolution))
            if not os.path.exists(hash_dir):
                os.makedirs(hash_dir)

            # Write the screen hashes to the directory for the resolution.
            hash_path = os.path.join(hash_dir, 'hashes.pickle')
            with open(hash_path, 'wb') as file:
                pickle.dump((hashes, labels), file, protocol=pickle.HIGHEST_PROTOCOL)

            # Close the game to prepare for loading the next resolution.
            menu.close_game()


if __name__ == '__main__':
    # Before running any of the following, make sure to turn off the Steam overlay.

    # Create image hashing datasets. Do not change the ordering.
    image_generator = ImageData()
    image_generator.make_dataset('level_select')
    image_generator.make_dataset('black')
    image_generator.make_dataset('blue')
    image_generator.make_dataset('blue_special')
    image_generator.make_dataset('counter')
    image_generator.make_dataset('column')
    image_generator.make_dataset('diagonal')
    image_generator.make_dataset('screen')

    # Create a training dataset of 2000 randomly generated levels.
    generator = LearningData()
    generator.make_dataset(1)
