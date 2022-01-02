import os
import json
import time
import pickle
import pyperclip
from subprocess import Popen

from grid import Cell
from main import GAMEIDS
from navigate import Navigator
from parse import GameParser, MenuParser, RESOLUTIONS
from solve import Solver

# Turn off steam overlay

class Generator:
    def __init__(self, steam_path=r'C:\Program Files (x86)\Steam', save_path='resources'):
        self.__options_path = os.path.join(steam_path, r'steamapps\common\{0}\saves\options.txt')
        self.__steam_path = os.path.join(steam_path, 'steam.exe')
        self._save_path = save_path

    def _load_game(self, game, resolution):
        # Close game if open already
        try:
            menu = Navigator()
            menu.close_game()
        except:
            pass

        options_path = self.__options_path.format(game)

        with open(options_path, 'r') as file:
            data = json.load(file)

        data['screenWidth'], data['screenHeight'] = resolution

        with open(options_path, 'w') as file:
            json.dump(data, file)

        Popen([self.__steam_path, r'steam://rungameid/'+GAMEIDS[game]],
               shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

        time.sleep(8)
        menu = Navigator()
        menu.window.move_mouse()
        menu.wait_until_loaded()

        return menu

class LevelData(Generator):
    def __init__(self, steam_path=r'C:\Program Files (x86)\Steam', save_path='resources/levels'):
        super().__init__(steam_path, save_path)

    def make_dataset(self, num_levels=2000):
        menu = self._load_game('Hexcells Infinite', (1920, 1080))

        menu_parser = MenuParser(menu.window)
        buttons = menu_parser.parse_main_menu()
        generator_button = buttons['level_generator']
        menu.window.click(generator_button)
        menu.window.move_mouse()
        time.sleep(2)

        levels = []
        labels = ['0'*(8-len(str(i)))+str(i) for i in range(num_levels)]
        for i in range(num_levels):
            pyperclip.copy(labels[i])

            play, _, seed = menu_parser.parse_generator()
            menu.window.click(seed)
            menu.window.move_mouse()
            time.sleep(0.2)
            menu.window.paste()
            time.sleep(0.2)

            menu.window.click(play)
            menu.window.move_mouse()
            menu_parser.wait_until_loaded()

            game_parser = GameParser(menu.window)
            solver = Solver(game_parser)

            grid = game_parser.parse_grid()
            grid_initial = game_parser.parse_grid()
            while True:
                left_click_cells, right_click_cells = solver.solve_single_step(grid, menu.window, None)
                if len(left_click_cells)+len(right_click_cells)-len(grid.unknown_cells()) == 0:
                    if len(left_click_cells) > 0:
                        game_parser.parse_clicked(grid, left_click_cells[1:], right_click_cells)
                        left_click_cells, right_click_cells = [left_click_cells[0]], []
                    else:
                        game_parser.parse_clicked(grid, left_click_cells, right_click_cells[1:])
                        left_click_cells, right_click_cells = [], [right_click_cells[0]]

                    game_parser.click_cells(left_click_cells, 'left')
                    game_parser.click_cells(right_click_cells, 'right')

                    menu.window.move_mouse()
                    time.sleep(1.2)
                    _, menu_button = menu_parser.parse_level_end()

                    menu.window.click(menu_button)
                    menu.window.move_mouse()
                    time.sleep(2)

                    play, _, _ = menu_parser.parse_generator()
                    menu.window.click(play)
                    menu.window.move_mouse()
                    menu_parser.wait_until_loaded()

                    game_parser.parse_clicked(grid, left_click_cells, right_click_cells)

                    menu.window.press_key('esc')
                    menu.window.move_mouse()
                    time.sleep(1.5)
                    _, _, exit_button = menu_parser.parse_level_exit()

                    menu.window.click(exit_button)
                    menu.window.move_mouse()
                    time.sleep(2)

                    levels.append((grid_initial, grid))
                    print('>>> {0}/{1}'.format(i+1, num_levels))
                    break

                _, remaining = game_parser.parse_clicked(grid, left_click_cells, right_click_cells)
                grid.remaining = remaining

        with open(os.path.join(self._save_path, 'levels.pickle'), 'wb') as file:
            pickle.dump((levels, labels), file, protocol=pickle.HIGHEST_PROTOCOL)

class ImageData(Generator):
    def __init__(self, steam_path=r'C:\Program Files (x86)\Steam', save_path='resources'):
        super().__init__(steam_path, save_path)

    def make_dataset(self, digit_type):
        hash_path = os.path.join(self._save_path, digit_type, 'hashes.pickle')
        game = 'Hexcells Infinite'
        if digit_type == 'blue_special':
            hash_path = os.path.join(self._save_path, 'blue', 'hashes.pickle')
            game = 'Hexcells Plus'
        
        hashes, labels = [], []
        for resolution in RESOLUTIONS:
            menu = self._load_game(game, resolution)

            if digit_type == 'column':
                hashes_res, labels_res = [], []
                for hint in ['normal', 'consecutive', 'non-consecutive']:
                    level_path = os.path.join(self._save_path, digit_type, '{}_level.hexcells'.format(hint))
                    menu.load_custom_level(level_path)

                    hashes_hint, labels_hint = self.__get_hashes(menu.window, digit_type+'_'+hint)
                    hashes_res += hashes_hint
                    labels_res += labels_hint

            elif digit_type == 'diagonal':
                hashes_res, labels_res = [], []
                for part in ['1', '2']:
                    for hint in ['normal', 'consecutive', 'non-consecutive']:
                        level_path = os.path.join(self._save_path, digit_type, '{0}_{1}_level.hexcells'.format(hint, part))
                        menu.load_custom_level(level_path)

                        hashes_hint, labels_hint = self.__get_hashes(menu.window, digit_type+'_{0}_{1}'.format(hint, part))
                        hashes_res += hashes_hint
                        labels_res += labels_hint

                        menu.back()
                        time.sleep(1)
                        menu.exit_level()

            else:
                level_path = os.path.join(self._save_path, digit_type, 'level.hexcells')

                if digit_type == 'level_select':
                    menu.load_save(1)

                elif digit_type == 'screen':
                    main_screen = menu.window.screenshot()

                    menu_parser = MenuParser(menu.window)
                    buttons = menu_parser.parse_main_menu()
                    save_slot_buttons = buttons['save_slots']
                    menu.window.click(save_slot_buttons[0])
                    menu.window.move_mouse()
                    menu_parser.wait_until_loaded()
                    level_select = menu.window.screenshot()

                    levels = menu_parser.parse_levels()
                    coords = levels['1-1']
                    menu.window.click(coords)
                    menu.window.move_mouse()
                    time.sleep(2)
                    menu.back()
                    level_exit = menu.window.screenshot()

                    menu.back()
                    game_parser = GameParser(menu.window)
                    solver = Solver(game_parser)
                    solver.solve('1-1', menu.window.title)

                    menu.window.move_mouse()
                    time.sleep(1.2)
                    level_end = menu.window.screenshot()
                    _, menu_button = menu_parser.parse_level_end()
                    menu.window.click(menu_button)
                    menu.window.move_mouse()
                    time.sleep(1)

                    menu.back()
                    buttons = menu_parser.parse_main_menu()
                    generator_button = buttons['level_generator']
                    menu.window.click(generator_button)
                    menu.window.move_mouse()
                    time.sleep(2)
                    level_generator = menu.window.screenshot()

                    menu.back()
                    buttons = menu_parser.parse_main_menu()
                    user_levels_button = buttons['user_levels']
                    menu.window.click(user_levels_button)
                    menu.window.move_mouse()
                    time.sleep(2)
                    user_levels = menu.window.screenshot()

                    menu.back()
                    buttons = menu_parser.parse_main_menu()
                    options_ubtton = buttons['options']
                    menu.window.click(options_button)
                    menu.window.move_mouse()
                    time.sleep(2)
                    options = menu.window.screenshot()

                    hashes = [main_screen, level_select, level_exit, level_end, level_generator, user_levels, options]
                    labels = ['main_menu', 'level_select', 'level_exit', 'level_end', 'level_generator', 'user_levels', 'options']

                    hash_path = os.path.join(self._save_path, digit_type, '{0}x{1}'.format(*resolution))
                    if not os.path.exists(hash_path):
                        os.makedirs(hash_path)

                    hash_path = os.path.join(hash_path, 'hashes.pickle')
                    with open(hash_path, 'wb') as file:
                        pickle.dump((hashes, labels), file, protocol=pickle.HIGHEST_PROTOCOL)

                    menu.close_game()
                    continue

                elif digit_type == 'blue_special':
                    menu.load_save(1)

                    menu_parser = MenuParser(menu.window)
                    levels = menu_parser.parse_levels()

                    menu.window.click(levels['4-6'])
                    menu.window.move_mouse()
                    menu.wait_until_loaded()

                else:
                    menu.load_custom_level(level_path)

                hashes_res, labels_res = self.__get_hashes(menu.window, digit_type)

            hashes += hashes_res
            labels += labels_res

            menu.close_game()

            # Only run for one resolution.
            if digit_type == 'counter':
                break

        if digit_type != 'screen':
            #if not delete_existing:
            #    with open(hash_path, 'rb') as file:
            #        existing_hashes, existing_labels = pickle.load(file)
            #        hashes += existing_hashes
            #        labels += existing_labels

            with open(hash_path, 'wb') as file:
                pickle.dump((hashes, labels), file, protocol=pickle.HIGHEST_PROTOCOL)

    def __get_hashes(self, window, digit_type):
        if digit_type == 'level_select':
            parser = MenuParser(window)
            labels = ['3-3', '3-2', '2-3', '2-2', '3-4', '3-1',
                      '2-4', '2-1', '3-5', '3-6', '2-5', '2-6',
                      '4-3', '4-2', '1-3', '1-2', '4-4', '4-1',
                      '1-4', '1-1', '4-5', '4-6', '1-5', '1-6',
                      '5-3', '5-2', '6-3', '6-2', '5-4', '5-1',
                      '6-4', '6-1', '5-5', '5-6', '6-5', '6-6']

            hashes = parser.parse_levels(training=True)

        elif digit_type in ['black', 'blue', 'blue_special', 'counter']:
            screenshot = window.screenshot()
            parser = GameParser(window)

            if digit_type == 'counter':
                remaining = 476
                labels = list(range(remaining, -1, -1))

                hashes = []
                for cell in parser.parse_cells(screenshot, Cell.ORANGE):
                    mistakes_hash, remaining_hash = parser.parse_counters(screenshot, training=True)

                    hashes.append(remaining_hash)

                    window.click_cell(cell, 'left')
                    remaining -= 1
                    screenshot = window.screenshot()

                hashes.append(mistakes_hash)

            elif digit_type == 'black':
                hashes = parser.parse_cells(screenshot, Cell.BLACK, training=True)
                labels = ['{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '1', '2', '3',
                          '4', '5', '6', '?', '-2-', '-3-', '-4-', '{0}', '0']

            elif digit_type == 'blue':
                hashes = parser.parse_cells(screenshot, Cell.BLUE, training=True)
                labels = [str(i) for i in range(0, 19)]

            elif digit_type == 'blue_special':
                hashes = parser.parse_cells(screenshot, Cell.BLUE, training=True)
                labels = ['{5}', '-2-', '5', '4', '2', '4', '5', '{4}', '{2}', '5', '3', '10', '10']

        else:
            if digit_type == 'column_normal':
                labels = [str(i) for i in range(0, 17)]

            elif digit_type == 'column_consecutive':
                labels = ['{'+str(i)+'}' for i in range(0, 17)]

            elif digit_type == 'column_non-consecutive':
                labels = ['-'+str(i)+'-' for i in range(2, 16)]

            elif digit_type == 'diagonal_normal_1':
                labels = [str(i) for i in range(1, 16)] + ['0', '0'] + [str(i) for i in range(15, 0, -1)]

            elif digit_type == 'diagonal_normal_2':
                labels = ['24', '25', '27', '26', '23', '22', '21', '20',
                          '19', '18', '17', '16', '16', '17', '18', '19',
                          '20', '23', '21', '24', '26', '27', '25', '22']

            elif digit_type == 'diagonal_consecutive_1':
                labels = ['{'+str(i)+'}' for i in range(1, 16)] + ['{0}', '{0}'] + ['{'+str(i)+'}' for i in range(15, 0, -1)]

            elif digit_type == 'diagonal_consecutive_2':
                labels = ['{24}', '{25}', '{27}', '{26}', '{23}', '{22}',
                          '{21}', '{20}', '{19}', '{18}', '{17}', '{16}',
                          '{16}', '{17}', '{18}', '{19}', '{20}', '{23}',
                          '{21}', '{24}', '{26}', '{27}', '{25}', '{22}']

            elif digit_type == 'diagonal_non-consecutive_1':
                labels = ['-'+str(i)+'-' for i in range(2, 16)] + ['-'+str(i)+'-' for i in range(15, 1, -1)]

            elif digit_type == 'diagonal_non-consecutive_2':
                labels = ['-23-', '-24-', '-27-', '-26-', '-25-', '-22-',
                          '-21-', '-20-', '-19-', '-18-', '-17-', '-16-',
                          '-16-', '-17-', '-18-', '-19-', '-20-', '-21-',
                          '-24-', '-25-', '-26-', '-22-', '-23-']
            else:
                raise RuntimeError('invalid digit type')

            parser = GameParser(window)
            hashes = parser.parse_grid(training=True)

        assert len(hashes) == len(labels)
        return hashes, labels

if __name__ == '__main__':
    level_generator = LevelData()
    level_generator.make_dataset()

    # Do not change the ordering.
    #image_generator = ImageData()
    #image_generator.make_dataset('level_select')
    #image_generator.make_dataset('screen')
    #image_generator.make_dataset('black')
    #image_generator.make_dataset('blue')
    #image_generator.make_dataset('blue_special')
    #image_generator.make_dataset('counter')
    #image_generator.make_dataset('column')
    #image_generator.make_dataset('diagonal')
