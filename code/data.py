import time, os, pickle

from subprocess import Popen

from navigate import Navigator
from window import ConfigWindow
from parse import MenuParser, GameParser
from grid import Cell

class Generator:
    __RESOLUTIONS =  ['2048x1152', '1920x1200', '1920x1080', '1680x1050', '1600x1200', '1600x900']
    __HEXCELLS_PATH = r'C:\Program Files (x86)\Steam\steamapps\content\app_304410\depot_304411\Hexcells Infinite.exe'

    def __init__(self, save_path='../resources'):
        self.__save_path = save_path

    def __reset_resolution(self):
        Popen([Generator.__HEXCELLS_PATH], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

        time.sleep(1)
        config_window = ConfigWindow()
        config_window.reset_resolution()

        menu = Navigator()
        menu.close_game()

    def make_dataset(self, digit_type):
        save_path = os.path.join(self.__save_path, digit_type, 'hashes.pickle')
        if digit_type == 'column':
            for i, hint in enumerate(['normal', 'consecutive', 'non-consecutive']):
                level_path = os.path.join(self.__save_path, digit_type, '{}_level.hexcells'.format(hint))
                self.__make_data_infinite(digit_type+'_'+hint, save_path, level_path, i==0)
        
        elif digit_type == 'diagonal':
            for part in ['1', '2']:
                for i, hint in enumerate(['normal', 'consecutive', 'non-consecutive']):
                    level_path = os.path.join(self.__save_path, digit_type, '{0}_{1}_level.hexcells'.format(hint, part))
                    self.__make_data_infinite(digit_type+'_{0}_{1}'.format(hint, part), save_path, level_path, part=='1' and i==0)
                    
        else:
            level_path = os.path.join(self.__save_path, digit_type, 'level.hexcells')
            self.__make_data_infinite(digit_type, save_path, level_path)
        
    def make_dataset_blue_special(self):
        game = Navigator()
        
        hashes, labels = [], []
        for i in range(6):
            screenshot = game.window.screenshot()
            parser = GameParser(game.window)
            
            hashes += parser.parse_cells(screenshot, Cell.BLUE, training=True)
            labels += ['{5}', '-2-', '5', '4', '2', '4', '5', '{4}', '{2}', '5', '3', '10', '10']
    
            assert len(hashes) == len(labels)
            print(i)
            time.sleep(3)

        game.close_game()
    
        hash_path = os.path.join(self.__save_path, 'blue', 'hashes.pickle')
        with open(hash_path, 'rb') as file:
            existing_hashes, existing_labels = pickle.load(file)
            hashes += existing_hashes
            labels += existing_labels

        with open(hash_path, 'wb') as file:
            pickle.dump((hashes, labels), file, protocol=pickle.HIGHEST_PROTOCOL)
        
    def __make_data_infinite(self, digit_type, hash_path, level_path, delete_existing=True): 
        self.__reset_resolution()

        hashes, labels = [], []
        for resolution in Generator.__RESOLUTIONS:
            Popen([Generator.__HEXCELLS_PATH], shell=True,
                  stdin=None, stdout=None, stderr=None, close_fds=True)

            time.sleep(1)

            config_window = ConfigWindow()
            config_window.start_game()

            time.sleep(1.5)

            menu = Navigator()
            
            if digit_type == 'level':
                menu.load_save_slot(1)
            else:
                menu.load_custom_level(level_path)
            
            menu.window.move_mouse()
            time.sleep(2.5)

            hashes_res, labels_res = self.__get_hashes(menu.window, digit_type, resolution)
            hashes += hashes_res
            labels += labels_res

            menu.close_game()
        
        if not delete_existing:
            with open(hash_path, 'rb') as file:
                existing_hashes, existing_labels = pickle.load(file)
                hashes += existing_hashes
                labels += existing_labels

        with open(hash_path, 'wb') as file:
            pickle.dump((hashes, labels), file, protocol=pickle.HIGHEST_PROTOCOL)

    def __get_hashes(self, window, digit_type, resolution):
        if digit_type == 'level':
            parser = MenuParser(window)
            labels = ['3-3', '3-2', '2-3', '2-2', '3-4', '3-1',
                      '2-4', '2-1', '3-5', '3-6', '2-5', '2-6',
                      '4-3', '4-2', '1-3', '1-2', '4-4', '4-1',
                      '1-4', '1-1', '4-5', '4-6', '1-5', '1-6',
                      '5-3', '5-2', '6-3', '6-2', '5-4', '5-1',
                      '6-4', '6-1', '5-5', '5-6', '6-5', '6-6']
            
            hashes = parser.parse_levels(training=True)
            
        elif digit_type in ['black', 'blue', 'counter']:
            screenshot = window.screenshot()
            parser = GameParser(window)
    
            if digit_type == 'counter':
                remaining = 478
                labels = list(range(remaining, -1, -1))
                
                hashes = []
                for cell in parser.parse_cells(screenshot, Cell.ORANGE_OLD):
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
    generator = Generator()
    #generator.make_dataset('level')
    #generator.make_dataset('black')
    #generator.make_dataset('blue')
    #generator.make_dataset_blue_special()
    #generator.make_dataset('counter')
    generator.make_dataset('column')
    generator.make_dataset('diagonal') 
