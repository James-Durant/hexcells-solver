import time, pyperclip

from window import get_window
from parse import MenuParser, GameParser
from solve import Solver

class Navigator:
    def __init__(self):
        self.__window = get_window()
        self.__menu_parser = MenuParser(self.__window)
        
        self.__menu_parser.parse_levels()

    @property
    def window(self):
        return self.__window

    def load_save_slot(self, slot, training=False):
        slots, generator = self.__menu_parser.parse_slots(training)
        
        if slot in [1,2,3]:
            self.__window.click(slots[slot-1])
            time.sleep(1)
        else:
            raise RuntimeError('invalid save slot given')

    def __solve(self, continuous):
        game_parser = GameParser(self.__window)
        solver = Solver(game_parser)
        solver.solve()
        
        self.__window.move_mouse()
        time.sleep(2)
        next_button, menu_button = self.__menu_parser.parse_level_end()

        if continuous and next_button is not None:
            self.__window.click(next_button)
            time.sleep(1.6)
            self.__solve()
        else:
            self.__window.click(menu_button)

    def solve_level(self, level):
        levels = self.__menu_parser.parse_levels()
        try:
            coords = levels[level]
        except KeyError:
            raise RuntimeError('invalid level given')
            
        self.__window.click(coords)
        self.__window.move_mouse()
        
        time.sleep(1.5)
        self.__solve(continuous=False)

    def solve_world(self, world):
        levels = self.__menu_parser.parse_levels()
        if world not in ['1', '2', '3', '4', '5', '6']:
            raise RuntimeError('world must be between 1-6 (inclusive)')
        
        self.__window.click(levels[world+'-1'])
        time.sleep(1.5)
        self.__solve(continuous=True)

    def solve_game(self):
        levels = self.__menu_parser.parse_levels()
        for world in ['1', '2', '3', '4', '5', '6']:
            self.__window.click(levels[world+'-6'])
            time.sleep(1.5)
            self.__solve(continuous=True)
            time.sleep(2)
     
    def back(self):
        self.__window.press_key('esc')
        
    def close_game(self):
        self.__window.close()
        
    def load_custom_level(self, level_path):
        with open(level_path, 'r') as file:
            level = file.read()
        pyperclip.copy(level)
        
        x1, y1, x2, y2 = self._get_position()
        x = round((x1+x2)*0.92)
        y = round((y1+y2)*0.97)
        
        self.__window.to_foreground()
        for i in range(5):
            self.__window.click((x, y))
