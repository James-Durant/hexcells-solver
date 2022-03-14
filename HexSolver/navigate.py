import time
import pyperclip

from solve import Solver
from window import get_window
from parse import MenuParser, LevelParser


class Navigator:
    """Provides an interface between high-level actions and menu parsing algorithms."""
    def __init__(self, use_level_hashes=True, use_screen_hashes=True):
        """Capture the active game window.

        Args:
            use_level_hashes (bool, optional): whether to load level perceptual hashes.
            use_screen_hashes (bool, optional): whether to load screen perceptual hashes.
        """
        self.__window = get_window()
        self.__menu_parser = MenuParser(self.__window, use_level_hashes, use_screen_hashes)

    @property
    def window(self):
        """
        Returns:
            window.Window: the active game window.
        """
        return self.__window

    @property
    def title(self):
        """
        Returns:
            str: name of the game that is currently active.
        """
        return self.__window.title

    def wait_until_loaded(self):
        """Wait until a menu screen has fully loaded before continuing."""
        self.__menu_parser.wait_until_loaded()

    def load_save(self, slot, screen=None):
        """Load a given save slot.

        Args:
            slot (int): save slot to load.
            screen (str, optional): screen that is currently showing.
        """
        # Check that the save slot is valid.
        if slot not in [1, 2, 3]:
            raise RuntimeError('Invalid save slot given')

        # Go from the current screen to the main menu screen.
        # Skip this if perceptual hashes are not being used (this is the case when called in data.py).
        if screen != 'main_menu':
            self.__transition_to_main_menu()

        # Parse the main menu screen to identify the save slot buttons.
        buttons = self.__menu_parser.parse_main_menu()
        save_slot_buttons = buttons['save_slots']

        # Click on the button corresponding to the chosen save slot.
        self.__window.click(save_slot_buttons[slot - 1])
        self.wait_until_loaded()

    def __transition_to_main_menu(self):
        """Transition from the current screen to the main menu."""
        # Determine what screen the game is currently showing.
        screen = self.__menu_parser.get_screen()

        # If in a level, exit it.
        # The new screen could be the level selection or the level generator screen.
        if screen in ['in_level', 'level_completion', 'level_exit']:
            self.exit_level(screen)

        # If not on the main menu screen, press escape to transition to it.
        if screen != 'main_menu':
            self.back()

    def exit_level(self, screen=None):
        """Exit a level to return to the level selection or the level generator screen.

        Args:
            screen (str, optional): screen that is currently showing.
        """
        # Determine what screen the game is currently showing if not given.
        if screen is None:
            screen = self.__menu_parser.get_screen()

        # If in a level, press escape to bring up the level exit screen.
        if screen == 'in_level':
            self.back()
            screen = 'level_exit'

        # Parse the level completion or level exit screen.
        if screen == 'level_completion':
            _, button = self.__menu_parser.parse_level_completion()

        elif screen == 'level_exit':
            _, _, button = self.__menu_parser.parse_level_exit()

        else:
            # Otherwise, the game is not currently in a level.
            return

        # Click on the button to exit the level.
        self.__window.click(button)
        self.wait_until_loaded()

    def __transition_to_level_select(self, save):
        """Transition to the level selection screen.

        Args:
            save (str): save slot to load.
        """
        # Get the screen currently shown.
        screen = self.__menu_parser.get_screen()

        # If the game is in a level, exit it.
        if screen in ['in_level', 'level_completion', 'level_exit']:
            self.exit_level(screen)
            # The new screen could be the level selection or level generator screen.
            screen = self.__menu_parser.get_screen()

        # If already on the level selection screen, nothing further needs to be done.
        if screen == 'level_select':
            return

        # If not on the main menu screen (level generator, user levels or options screen),
        # press the escape key to move to the main menu screen.
        if screen != 'main_menu':
            self.back()

        # Load the given save to move to the level selection screen.
        self.load_save(int(save) if save != '-' else 1)

    def __transition_to_level_generator(self):
        """Transition to the level generator screen."""
        # Get the screen that is currently being shown.
        screen = self.__menu_parser.get_screen()

        # If the game is in a level, exit it.
        if screen in ['in_level', 'level_completion', 'level_exit']:
            self.exit_level(screen)
            # The new screen could be the level selection or level generator screen.
            screen = self.__menu_parser.get_screen()

        # If already on the level selection screen, nothing further needs to be done.
        if screen == 'level_generator':
            return

        # Otherwise, move to the main menu screen, parse it and click on the level generator button.
        self.__transition_to_main_menu()
        buttons = self.__menu_parser.parse_main_menu()
        generator_button = buttons['level_generator']
        self.__window.click(generator_button)
        self.wait_until_loaded()

    def solve(self, continuous, level=None, delay=False):
        """Solve a single level using the solver. This method assumes the level is being displayed.

        Args:
            continuous (bool): whether to click the "next level" button on completion.
            level (str, optional): level being solved.
            delay (bool, optional): whether to add a delay after clicking cells.
        """
        # Solve the level.
        game_parser = LevelParser(self.__window)
        solver = Solver(game_parser)
        solver.solve(self.__window.title, level, delay)

        # Wait for the level completion screen to load and parse it.
        self.__window.move_mouse()
        time.sleep(1.25)
        next_button, menu_button = self.__menu_parser.parse_level_completion()

        # Check if more levels are to be solved and that the level was not the last in a set.
        if continuous and next_button is not None:
            self.__window.click(next_button)
            time.sleep(1.75)

            # If a level was given, update it to the next in the set.
            if level is not None:
                level = level[:-1] + str(int(level[-1]) + 1)

            # Run this method recursively on the next level.
            self.solve(continuous, level, delay)

        else:
            # Otherwise, return to the level selection screen.
            self.__window.click(menu_button)

    def solve_level(self, save, level_str, delay=False):
        """Solve a single level. This method does not assume that the level is being displayed.

        Args:
            save (str): save slot to load.
            level_str (str): level to solve.
            delay (bool, optional): whether to use a delay after clicking cells.
        """
        # Move to the level selection screen.
        self.__transition_to_level_select(save)

        # Parse the level selection screen.
        levels = self.__menu_parser.parse_level_selection()
        try:
            # Try to click on the button corresponding to the chosen level.
            self.__window.click(levels[level_str])

        except KeyError:
            # If the level's button was not parsed, the level has not been unlocked.
            raise RuntimeError('Selected level not unlocked yet')

        # Wait until the level has loaded and then solve it.
        self.wait_until_loaded()
        self.solve(False, level_str, delay)  # False to only solve the one level.

    def solve_set(self, save, set_str, delay=False):
        """Solve a set of levels.

        Args:
            save (str): save slot to load
            set_str (str): set of levels to solve.
            delay (bool, optional): whether to add a delay after clicking cells.
        """
        # Move to the level selection screen.
        self.__transition_to_level_select(save)

        # Parse the level selection screen.
        levels = self.__menu_parser.parse_level_selection()

        try:
            # Try to click on the button corresponding to the first level in the set.
            level = set_str + '-1'
            self.__window.click(levels[level])

        except KeyError:
            # If the button was not parsed, the level has not been unlocked.
            raise RuntimeError('Selected level not unlocked yet')

        # Wait until the level has loaded and then solve it.
        self.wait_until_loaded()
        self.solve(True, level, delay)  # True to solve the rest of the levels in the set.

    def solve_game(self, save, delay=False):
        """Solves all levels in the game currently running.

        Args:
            save (str): save slot to load.
            delay (bool, optional): whether to add a delay after clicking cells.
        """
        # Move to the level selection screen.
        self.__transition_to_level_select(save)

        # Solve each set in the game.
        for set_str in ['1', '2', '3', '4', '5', '6']:
            self.solve_set(save, set_str, delay)
            self.wait_until_loaded()

    def level_generator(self, num_levels, delay=False, train=None):
        """Run the solver or a machine learning model on a randomly generated level.

        Args:
            num_levels (int): number of levels to generate.
            delay (bool, optional): whether to use a delay after clicking cells.
            train (function, optional): function to call to run a model.
        """
        # Check that the game is Hexcells Infinite as the other two games do not have the level generator.
        if self.__window.title != 'Hexcells Infinite':
            raise RuntimeError('Only Hexcells Infinite has the level generator')

        # Transition to the level generator screen.
        self.__transition_to_level_generator()

        try:
            # Iterate for the given number of levels.
            agent = None
            for _ in range(num_levels):
                # Parse the level generation screen.
                buttons = self.__menu_parser.parse_generator()
                play_button, random_button = buttons['play'], buttons['random']

                # Click on the button to input a new random seed.
                self.__window.click(random_button, move_mouse=False)

                # Click on the button to generate the level.
                self.__window.click(play_button)
                self.wait_until_loaded()  # Wait for the level to be generated.

                # Run a model if one was given.
                if train:
                    agent = train(agent)
                    # Exit the level after it has been solved.
                    self.__window.move_mouse()
                    time.sleep(1.5)
                    self.exit_level()
                else:
                    # Otherwise, run the solver.
                    self.solve(False, delay=delay)

        except KeyboardInterrupt:
            # Stop solving on Control + C event.
            return

    def load_custom_level(self, level_path, screen=None):
        """Load a custom level from a given file path.

        Args:
            level_path (str): path to the custom level to load.
            screen (str, optional): screen that is currently showing.
        """
        # Go from the current screen to the main menu.
        if screen != 'main_menu':
            self.__transition_to_main_menu()

        # Load the given level file and copy its contents to the clipboard.
        with open(level_path, 'r') as file:
            level = file.read()
            pyperclip.copy(level)

        # Parse the main menu screen and click on the user levels button.
        # With the level contents in the clipboard, this will load the custom level.
        buttons = self.__menu_parser.parse_main_menu()
        user_level_button = buttons['user_levels']
        self.__window.click(user_level_button)
        self.__menu_parser.wait_until_loaded()

    def back(self):
        """Press the escape key and wait for a short period."""
        self.__window.press_key('esc')
        time.sleep(1.5)

    def close_game(self):
        """Close the active game window."""
        self.__window.close()
