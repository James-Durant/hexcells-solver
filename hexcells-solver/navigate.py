import sys
import time
import pyperclip

from solve import Solver
from window import get_window
from parse import MenuParser, GameParser


class Navigator:
    def __init__(self, training=False):
        self.__window = get_window()
        self.__menu_parser = MenuParser(self.__window, training=training)

    @property
    def window(self):
        return self.__window

    @property
    def title(self):
        return self.__window.title

    def wait_until_loaded(self):
        self.__menu_parser.wait_until_loaded()

    def load_save(self, slot):
        self.__transition_to_main_menu()
        buttons = self.__menu_parser.parse_main_menu()
        save_slot_buttons = buttons['save_slots']

        if slot in [1, 2, 3]:
            self.__window.click(save_slot_buttons[slot - 1])
            self.__window.move_mouse()
            self.wait_until_loaded()
        else:
            raise RuntimeError('invalid save given')

    def __transition_to_main_menu(self):
        screen = self.__menu_parser.get_screen()

        if screen == 'in_level':
            self.back()
            time.sleep(0.5)
            self.exit_level()

        elif screen == 'level_completion':
            _, menu_button = self.__menu_parser.parse_level_end()
            self.__window.click(menu_button)
            self.__window.move_mouse()
            time.sleep(1.6)
            self.back()

        elif screen == 'level_exit':
            self.exit_level()

        if screen != 'main_menu':
            self.back()

    def exit_level(self):
        exit_button = None
        while True:
            try:
                _, _, exit_button = self.__menu_parser.parse_level_exit()
                break
            except KeyboardInterrupt:
                sys.exit()
                pass
            except Exception:
                continue

        self.__window.click(exit_button)
        self.__window.move_mouse()
        self.wait_until_loaded()

    def __transition_to_level_select(self, save):
        screen = self.__menu_parser.get_screen()

        if screen == 'level_select':
            pass

        elif screen == 'in_level':
            self.back()
            time.sleep(0.5)
            self.exit_level()

        elif screen == 'level_completion':
            _, menu_button = self.__menu_parser.parse_level_end()
            self.__window.click(menu_button)
            self.__window.move_mouse()
            self.wait_until_loaded()

        elif screen == 'level_exit':
            self.exit_level()

        else:
            if screen != 'main_menu':
                self.back()
            
            self.load_save(int(save) if save != '-' else 1)

    def solve(self, continuous, level=None, delay=False):
        game_parser = GameParser(self.__window)
        solver = Solver(game_parser)
        solver.solve(level, self.__window.title, delay)

        self.__window.move_mouse()
        time.sleep(1.2)
        next_button, menu_button = self.__menu_parser.parse_level_end()

        if continuous and next_button is not None:
            self.__window.click(next_button)
            self.__window.move_mouse()
            time.sleep(1.6)

            if level is not None:
                level = level[:-1] + str(int(level[-1]) + 1)
            self.solve(continuous, level, delay)
        else:
            self.__window.click(menu_button)
            self.__window.move_mouse()

    def solve_level(self, save, level_str, delay=False):
        self.__transition_to_level_select(save)

        levels = self.__menu_parser.parse_levels()
        try:
            coords = levels[level_str]
        except KeyError:
            raise RuntimeError('Selected level is not unlocked yet')

        self.__window.click(coords)
        self.__window.move_mouse()
        self.wait_until_loaded()

        self.solve(False, level_str, delay)

    def solve_set(self, save, set_str, delay=False):
        self.__transition_to_level_select(save)

        levels = self.__menu_parser.parse_levels()
        if set_str not in ['1', '2', '3', '4', '5', '6']:
            raise RuntimeError('Set must be between 1-6 (inclusive)')
        
        try:
            level = set_str + '-1'
            self.__window.click(levels[level])
        except KeyError:
            raise RuntimeError('Selected level is not unlocked yet')
            
        self.__window.move_mouse()
        self.wait_until_loaded()
        self.solve(True, level, delay)

    def solve_game(self, save, delay=False):
        self.__transition_to_level_select(save)

        for set_str in ['1', '2', '3', '4', '5', '6']:
            self.solve_set(save, set_str, delay)
            self.wait_until_loaded()

    def level_generator(self, func=None):
        if self.__window.title != 'Hexcells Infinite':
            raise RuntimeError('Only Hexcells Infinite has level generator')

        screen = self.__menu_parser.get_screen()
        if screen != 'level_generator':
            self.__transition_to_main_menu()

            buttons = self.__menu_parser.parse_main_menu()
            generator_button = buttons['level_generator']
            self.__window.click(generator_button)
            self.wait_until_loaded()

        while True:
            buttons = self.__menu_parser.parse_generator()
            play_button, random_button = buttons['play'], buttons['random']
            self.__window.click(random_button)
            self.__window.click(play_button)

            self.__window.move_mouse()
            self.wait_until_loaded()
            if func:
                func()
                # make this code better
                self.__window.move_mouse()
                time.sleep(1.2)
                next_button, menu_button = self.__menu_parser.parse_level_end()
                self.__window.click(menu_button)
                self.__window.move_mouse()
                time.sleep(1.2)
            else:
                self.solve(False)

    def back(self):
        self.__window.press_key('esc')
        time.sleep(1.5)

    def close_game(self):
        self.__window.close()

    def load_custom_level(self, level_path):
        self.__transition_to_main_menu()

        with open(level_path, 'r') as file:
            level = file.read()

        pyperclip.copy(level)

        buttons = self.__menu_parser.parse_main_menu()
        user_level_button = buttons['user_levels']
        self.__window.click(user_level_button)
        self.__window.move_mouse()
        self.__menu_parser.wait_until_loaded()
