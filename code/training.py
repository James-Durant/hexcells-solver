from grid import Cell
import os

def make_counter_dataset(parser, save_path, remaining=478):
    if not os.path.exists(save_path):
        os.makedir(save_path)
    
    image = parser.window.screenshot()
    #Load the counter level: all cells are unknown and blue.
    for cell in parser.parse_cells_of_colour(image, Cell.ORANGE_OLD):
        file_path = save_path + '/counter_{}.png'.format(remaining)
        parser.parse_counters(image, save_path=file_path)
        parser.window.click_cell(cell, 'left')
        remaining -= 1
        image = parser.window.screenshot()
            
def make_blue_digit_dataset(parser, save_path):
    if not os.path.exists(save_path):
        os.makedir(save_path)
    
    image = parser.window.screenshot()
    #Load the blue digit level which has all blue cells with digits from 0-18.
    #Cells will be parsed in order from 0 to 18 by the way the level is defined.
    save_paths = [save_path+'/blue_digit_{}.png'.format(digit) for digit in range(0,19)]
    parser.parse_cells_of_colour(image, Cell.BLUE, save_paths)

            
def make_black_digit_dataset(parser, save_path):
    if not os.path.exists(save_path):
        os.makedir(save_path)
        
    digits = ['{1}', '{2}', '{3}', '{4}', '{5}', '{6}',
               '1',   '2',   '3',   '4',   '5',   '6',
              '-2-', '-3-', '-4-',  '?',  '{0}',  '0'] 
    
    image = parser.window.screenshot()
    #Load the blue digit level which has all blue cells with digits from 0-18.
    #Cells will be parsed in order from 0 to 18 by the way the level is defined.
    save_paths = [save_path+'/black_digit_{}.png'.format(digit) for digit in digits]
    parser.parse_cells_of_colour(image, Cell.BLACK, save_paths)