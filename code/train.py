import time, h5py
import numpy as np

from subprocess import Popen

from window import ConfigWindow, GameWindow
from parse import Parser
from grid import Cell
        
class Generator:
    __RESOLUTIONS =  ['2048x1152', '1920x1200', '1920x1080', '1680x1050', 
                      '1600x1200', '1600x900',  '1280x720',  '1152x864']
    __HEXCELLS_PATH = r'C:\Program Files (x86)\Steam\steamapps\content\app_304410\depot_304411\Hexcells Infinite.exe'
    
    def __init__(self, save_path='../resources/training'):
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
        self.__reset_resolution()
        
        h5_file = h5py.File(self.__save_path+'/{}.h5'.format(digit_type), 'w')

        for resolution in [Generator.__RESOLUTIONS[0]]:
            Popen([Generator.__HEXCELLS_PATH], shell=True, 
                  stdin=None, stdout=None, stderr=None, close_fds=True)
            
            time.sleep(1)
            
            config_window = ConfigWindow()
            config_window.load_game()
                    
            time.sleep(1)
 
            level_path = self.__save_path+'/{}_level.hexcells'.format(digit_type)
            game_window = GameWindow('Hexcells Infinite')
            game_window.load_custom_level(level_path)
            game_window.move_mouse()
            
            time.sleep(2.5)
            
            self.__parse_digits(game_window, h5_file, resolution, digit_type)
            game_window.close()

        h5_file.close()
        
    def __parse_digits(self, game_window, h5_file, resolution, digit_type):
        screenshot = game_window.screenshot()
        parser = Parser(game_window)
        
        if digit_type == 'counter':
            remaining = 478
            images = []
            digits = [str(x) for x in range(remaining, -1, -1)]
            for cell in parser.parse_cells(screenshot, Cell.ORANGE_OLD):
                mistakes_img, remaining_img = parser.parse_counters(screenshot, training=True)
                images.append(remaining_img)
                game_window.click_cell(cell, 'left')
                remaining -= 1
                screenshot = game_window.screenshot()
            images.append(mistakes_img)
                
        elif digit_type == 'black':
            images = parser.parse_cells(screenshot, Cell.BLACK, training=True)
            digits = ['{1}', '{2}', '{3}', '{4}', '{5}', '{6}', 
                      '1', '2', '3', '4', '5', '6', 
                      '?', '-2-', '-3-', '-4-', '{0}', '0']
            
        elif digit_type == 'blue':
            images = parser.parse_cells(screenshot, Cell.BLUE, training=True)
            digits = [str(x) for x in range(0, 19)]
        else:
            raise RuntimeError('invalid digit type')

        labels = [label.encode('ascii') for label in digits]
        
        max_x = max(image.shape[0] for image in images)
        max_y = max(image.shape[1] for image in images)
        images = [self.__pad_to_dims(image, max_x, max_y) for image in images]
        
        h5_file.create_group(resolution)
        h5_file.create_dataset('{}/images'.format(resolution), data=images, dtype=np.uint8)
        h5_file.create_dataset('{}/labels'.format(resolution), data=labels)

    def __pad_to_dims(self, arr, xx, yy):
        h = arr.shape[0]
        w = arr.shape[1]
    
        a = (xx - h) // 2
        aa = xx - a - h
    
        b = (yy - w) // 2
        bb = yy - b - w
    
        return np.pad(arr, pad_width=((a, aa), (b, bb)), mode='constant', constant_values=255)

if __name__ == "__main__":
    generator = Generator()
    #generator.make_dataset('black')
    #generator.make_dataset('blue')
    generator.make_dataset('counter')
    
"""    
def make_grid_normal_diag_dataset(parser, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    digits = ['0_left',   '0_right',  '1_left',   '16_right', '2_left',   '3_left',   '15_right',
              '4_left',   '5_left',   '14_right', '6_left',   '7_left',   '13_right', '8_left',
              '9_left',   '12_right', '10_left',  '11_left',  '11_right', '12_left',  '13_left',
              '10_right', '14_left',  '15_left',  '9_right',  '16_left',  '8_right',  '7_right',
              '6_right',  '5_right',  '4_right',  '3_right',  '2_right',  '1_right']
    
    save_paths = [save_path+'/grid_{}.png'.format(digit) for digit in digits] 
    parser.parse_grid(save_paths)

def make_grid_consecutive_diag_dataset(parser, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    digits = ['{0}_left',   '{0}_right',  '{1}_left',   '{16}_right', '{2}_left',   '{3}_left',   '{15}_right',
              '{4}_left',   '{5}_left',   '{14}_right', '{6}_left',   '{7}_left',   '{13}_right', '{8}_left',
              '{9}_left',   '{12}_right', '{10}_left',  '{11}_left',  '{11}_right', '{12}_left',  '{13}_left',
              '{10}_right', '{14}_left',  '{15}_left',  '{9}_right',  '{16}_left',  '{8}_right',  '{7}_right',
              '{6}_right',  '{5}_right',  '{4}_right',  '{3}_right',  '{2}_right',  '{1}_right']
    
    save_paths = [save_path+'/grid_{}.png'.format(digit) for digit in digits] 
    parser.parse_grid(save_paths)
"""
    