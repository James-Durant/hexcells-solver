import time, pyperclip

from window import get_window
from parse import MenuParser, GameParser
from solve import Solver

class Navigator:
    def __init__(self):
        self.__window = get_window()
        self.__menu_parser = MenuParser(self.__window)
        self.__status = 'Menu Screen'

    @property
    def window(self):
        return self.__window
    
    @property
    def status(self):
        return self.__status
    
    def save_slot(self, slot, training=False):
        slots, _ = self.__menu_parser.parse_slots(training)
        
        if slot in [1,2,3]:
            self.__window.click(slots[slot-1])
            time.sleep(1)
        else:
            raise RuntimeError('invalid save slot given')

    def puzzle_generator(self, func=None):
        if self.__window.title != 'Hexcells Infinite':
            raise RuntimeError('Only Hexcells Infinite has puzzle generator')
        
        #_, generator = self.__menu_parser.parse_slots()
        #self.__window.click(generator)
        
        while True:
            time.sleep(1)
            play, random = self.__menu_parser.parse_generator()
            self.__window.click(random)
            self.__window.click(play)
        
            self.__window.move_mouse()
            self.__menu_parser.wait_until_loaded()
            if func:
                func()
            else:
                self.solve(False)

    def solve(self, continuous, level=None):
        game_parser = GameParser(self.__window)
        solver = Solver(game_parser)
        solver.solve(level, self.__window.title)
        
        self.__window.move_mouse()
        time.sleep(1.2)
        next_button, menu_button = self.__menu_parser.parse_level_end()

        if continuous and next_button is not None:
            self.__window.click(next_button)
            self.__window.move_mouse()
            time.sleep(1.6)
            
            if level is not None:
                level = level[:-1] + str(int(level[-1])+1)
            self.solve(continuous, level)
        else:
            self.__window.click(menu_button)

    def load_level(self, level):
        levels = self.__menu_parser.parse_levels()
        try:
            coords = levels[level]
        except KeyError:
            raise RuntimeError('invalid level given')
            
        self.__window.click(coords)
        self.__window.move_mouse()
        self.__menu_parser.wait_until_loaded()
        
    def solve_level(self, level_str):
        self.load_level(level_str)
        self.solve(False, level_str)

    def solve_set(self, set_str):
        levels = self.__menu_parser.parse_levels()
        if set_str not in ['1', '2', '3', '4', '5', '6']:
            raise RuntimeError('world must be between 1-6 (inclusive)')
        
        level = set_str+'-1'
        self.__window.click(levels[level])
        self.__window.move_mouse()
        time.sleep(1.5)
        self.solve(True, level)

    def solve_game(self):
        for world in ['1', '2', '3', '4', '5', '6']:
            self.solve_world(world)
            time.sleep(2)
     
    def back(self):
        self.__window.press_key('esc')
        
    def close_game(self):
        self.__window.close()
        
    def load_custom_level(self, level_path):
        with open(level_path, 'r') as file:
            level = file.read()
        pyperclip.copy(level)
        
        x1, y1, x2, y2 = self.__window._get_position()
        x = round((x1+x2)*0.92)
        y = round((y1+y2)*0.97)
        
        self.__window.to_foreground()
        for i in range(5):
            self.__window.click((x-x1, y-y1))
