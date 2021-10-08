import json, os, time, pickle
from subprocess import Popen

from grid import Cell
from main import GAMEIDS
from navigate import Navigator
from parse import GameParser, MenuParser

# Turn off steam overlay

class Generator:
    __RESOLUTIONS = [(2560, 1920), (2560, 1600), (2048, 1152),
                     (1920, 1440), (1920, 1200), (1920, 1080),
                     (1680, 1050), (1600, 1200)]

    def __init__(self, steam_path=r'C:\Program Files (x86)\Steam', save_path='resources'):
        self.__options_path = os.path.join(steam_path, r'steamapps\common\Hexcells Infinite\saves\options.txt')
        self.__steam_path = os.path.join(steam_path, 'steam.exe')
        self.__save_path = save_path

    def __load_game(self, game, resolution):
        # Close game if open already
        try:
            menu = Navigator()
            menu.close_game()
        except:
            pass
        
        with open(self.__options_path, 'r') as file:
            data = json.load(file)
            
        data['screenWidth'], data['screenHeight'] = resolution

        with open(self.__options_path, 'w') as file:
            json.dump(data, file)
        
        Popen([self.__steam_path, r'steam://rungameid/'+GAMEIDS[game]],
               shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
        
        time.sleep(5)
        menu = Navigator()
        menu.window.move_mouse()
        menu.wait_until_loaded()
        
        return menu
        
    def make_dataset(self, digit_type):         
        hashes, labels = [], []
        for resolution in Generator.__RESOLUTIONS:
            if digit_type == 'blue_special':
                menu = self.__load_game('Hexcells Plus', resolution)
                hash_path = os.path.join(self.__save_path, 'blue', 'hashes.pickle')
                delete_existing = False
                
            else:
                menu = self.__load_game('Hexcells Infinite', resolution)
                hash_path = os.path.join(self.__save_path, digit_type, 'hashes.pickle')
                delete_existing = True
            
            if digit_type == 'column':
                hashes_res, labels_res = [], []
                for hint in ['normal', 'consecutive', 'non-consecutive']:
                    level_path = os.path.join(self.__save_path, digit_type, '{}_level.hexcells'.format(hint))
                    menu.load_custom_level(level_path)
                    
                    hashes_hint, labels_hint = self.__get_hashes(menu.window, digit_type+'_'+hint)
                    hashes_res += hashes_hint
                    labels_res += labels_hint
                
            elif digit_type == 'diagonal':
                hashes_res, labels_res = [], []
                for part in ['1', '2']:
                    for hint in ['normal', 'consecutive', 'non-consecutive']:
                        level_path = os.path.join(self.__save_path, digit_type, '{0}_{1}_level.hexcells'.format(hint, part))
                        menu.load_custom_level(level_path)
                        
                        hashes_hint, labels_hint = self.__get_hashes(menu.window, digit_type+'_{0}_{1}'.format(hint, part))
                        hashes_res += hashes_hint
                        labels_res += labels_hint
                    
                        menu.back()
                        time.sleep(1)
                        menu.exit_level()
                        
            else:
                level_path = os.path.join(self.__save_path, digit_type, 'level.hexcells')
                
                if digit_type == 'level_select':
                    menu.load_save(1)
                
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
        
        if not delete_existing:
            with open(hash_path, 'rb') as file:
                existing_hashes, existing_labels = pickle.load(file)
                hashes += existing_hashes
                labels += existing_labels

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
    generator = Generator()
    # Do not change the ordering.
    #generator.make_dataset('level_select')
    #generator.make_dataset('black')
    #generator.make_dataset('blue')
    generator.make_dataset('blue_special')
    #generator.make_dataset('counter')
    #generator.make_dataset('column')
    #generator.make_dataset('diagonal') 
