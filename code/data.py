import time, os, pickle

from subprocess import Popen

from window import ConfigWindow, GameWindow
from parse import Parser
from grid import Cell

class Generator:
    __RESOLUTIONS =  ['2048x1152', '1920x1200', '1920x1080', '1680x1050',
                      '1600x1200', '1600x900',  '1280x720',  '1152x864']
    __HEXCELLS_PATH = r'C:\Program Files (x86)\Steam\steamapps\content\app_304410\depot_304411\Hexcells Infinite.exe'

    def __init__(self, save_path='../resources'):
        self.__save_path = save_path

    def __reset_resolution(self):
        Popen([Generator.__HEXCELLS_PATH], shell=True,
               stdin=None, stdout=None, stderr=None, close_fds=True)

        time.sleep(1)
        config_window = ConfigWindow()
        config_window.reset_resolution()

        game_window = GameWindow('Hexcells Infinite')
        game_window.close()

    def make_dataset(self, digit_type):
        save_path = os.path.join(self.__save_path, digit_type)
        self.__reset_resolution()

        hashes, labels = [], []
        for resolution in Generator.__RESOLUTIONS:
            Popen([Generator.__HEXCELLS_PATH], shell=True,
                  stdin=None, stdout=None, stderr=None, close_fds=True)

            time.sleep(1)

            config_window = ConfigWindow()
            config_window.load_game()

            time.sleep(1)

            level_path = os.path.join(save_path, 'level.hexcells')
            game_window = GameWindow('Hexcells Infinite')
            game_window.load_custom_level(level_path)
            game_window.move_mouse()

            time.sleep(2.5)

            hashes_res, labels_res = self.__parse_digits(game_window, digit_type, resolution)
            hashes += hashes_res
            labels += labels_res

            game_window.close()
        
        with open(os.path.join(save_path, 'hashes.pickle'), 'wb') as file:
            pickle.dump((hashes, labels), file, protocol=pickle.HIGHEST_PROTOCOL)

    def __parse_digits(self, game_window, digit_type, resolution):
        screenshot = game_window.screenshot()
        parser = Parser(game_window, training=True)

        hashes = []
        if digit_type == 'counter':
            remaining = 478
            labels = list(range(remaining, -1, -1))
            for cell in parser.parse_cells(screenshot, Cell.ORANGE_OLD):
                mistakes_hash, remaining_hash = parser.parse_counters(screenshot, training=True)

                hashes.append(remaining_hash)

                game_window.click_cell(cell, 'left')
                remaining -= 1
                screenshot = game_window.screenshot()

            hashes.append(mistakes_hash)

        elif digit_type == 'black':
            hashes = parser.parse_cells(screenshot, Cell.BLACK, training=True)
            labels = ['{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '1', '2', '3', 
                      '4', '5', '6', '?', '-2-', '-3-', '-4-', '{0}', '0']

        elif digit_type == 'blue':
            hashes = parser.parse_cells(screenshot, Cell.BLUE, training=True)
            labels = list(range(0, 19))

        else:
            if digit_type == 'column':
                labels = ['{16}', '-15-', '16', '{15}', '-14-', '15', '14',
                          '{14}', '-13-', '-12-', '-11-', '{12}', '-10-',
                           '13', '{13}', '-9-', '12', '-8-', '-7-', '{6}', 
                          '{11}', '-6-', '9', '{9}', '6', '-5-', '11', '-4-',
                          '-3-', '{5}', '-2-', '8',  '{8}', '5', '0', '{0}',
                          '{4}', '7', '{7}', '4', '{3}', '3', '{2}', '2',
                          '{1}', '1']
    
            elif digit_type == 'diag_normal':
                labels = ['16', '0', '0', '1', '2', '15', '3', '4', '14', '5', 
                          '6', '13', '7', '8', '12', '9', '10', '11', '11', 
                          '12', '10', '13', '14', '15', '9', '16', '8', '7', 
                          '6', '5', '4', '3', '2', '1']
                
            elif digit_type == 'diag_consecutive':
                labels = ['{16}', '{0}', '{0}', '{1}', '{2}', '{15}', '{3}', 
                          '{4}', '{14}', '{5}',  '{6}', '{13}', '{7}',  '{8}', 
                          '{12}', '{9}',  '{10}', '{11}', '{11}', '{12}', 
                          '{13}', '{10}', '{14}', '{15}', '{9}', '{16}', '{8}', 
                          '{7}', '{6}', '{5}', '{4}', '{3}', '{2}', '{1}']
                
            elif digit_type == 'diag_non-consecutive':
                labels = ['-16-', '-2-', '-3-', '-15-', '-4-', '-5-', '-14-',
                          '-6-', '-7-', '-13-', '-8-', '-9-', '-10-', '-12-',
                          '-11-', '-12-', '-11-', '-13-', '-14-', '-10-',
                          '-15-', '-16-', '-9-', '-8-', '-7-', '-6-', '-5-',
                          '-4-', '-3-', '-2-']
                
            else:
                raise RuntimeError('invalid digit type')
                
            parser = Parser(game_window, training=False)
            hashes = parser.parse_grid(training=True)
            
        return hashes, labels

if __name__ == "__main__":
    generator = Generator()
    #generator.make_dataset('black')
    #generator.make_dataset('blue')
    #generator.make_dataset('counter')
    #generator.make_dataset('column')
    #generator.make_dataset('diag_normal')
    #generator.make_dataset('diag_consecutive')
    #generator.make_dataset('diag_non-consecutive')
