import time, pyperclip

from window import get_window
from parse import MenuParser, GameParser
from solve import Solver

class Navigator:
    def __init__(self):
        self.__window = get_window()
        self.__menu_parser = MenuParser(self.__window)

    @property
    def window(self):
        return self.__window
    
    @property
    def title(self):
        return self.__window.title
    
    def wait_until_loaded(self):
        self.__menu_parser.wait_until_loaded()
    
    def load_save(self, save, training=False):
        self.__transition_to_main_menu()
        saves, _ = self.__menu_parser.parse_slots(training)
        
        if save in [1,2,3]:
            self.__window.click(saves[save-1])
            self.__window.move_mouse()
            self.__menu_parser.wait_until_loaded()
            time.sleep(0.25)
        else:
            raise RuntimeError('invalid save given')

    def __transition_to_main_menu(self):
        screen = self.__menu_parser.get_screen() 
        
        if screen == 'in_level':
            self.back()
            time.sleep(0.5)
            self.__exit_level()
        
        elif screen == 'level_exit':
            self.__exit_level()
            
        if screen != 'main_menu':
            self.back()

    def __exit_level(self):
        _, _, exit_button = self.__menu_parser.parse_level_exit()
        self.__window.click(exit_button)
        self.__window.move_mouse()
        self.__menu_parser.wait_until_loaded()
        
    def __transition_to_level_select(self, save):
        screen = self.__menu_parser.get_screen() 
    
        if screen == 'level_select':
            pass   
        
        elif screen == 'in_level':
            self.back()
            time.sleep(0.5)
            self.__exit_level()
        
        elif screen == 'level_exit':
            self.__exit_level()
            
        else:
            if screen != 'main_menu':
                self.back()

            self.load_save(int(save) if save != '-' else 1)

    def level_generator(self, func=None):
        if self.__window.title != 'Hexcells Infinite':
            raise RuntimeError('Only Hexcells Infinite has level generator')
            
        screen = self.__menu_parser.get_screen() 
        if screen != 'level_generator':
            self.__transition_to_main_menu()

            _, generator = self.__menu_parser.parse_slots()
            self.__window.click(generator)
            self.__menu_parser.wait_until_loaded()
        
        while True:
            play, random = self.__menu_parser.parse_generator()
            self.__window.click(random)
            self.__window.click(play)
        
            self.__window.move_mouse()
            self.__menu_parser.wait_until_loaded()
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
            self.__window.move_mouse()

    def solve_level(self, save, level_str):
        self.__transition_to_level_select(save)
        
        levels = self.__menu_parser.parse_levels()
        try:
            coords = levels[level_str]
        except KeyError:
            raise RuntimeError('invalid level given')
            
        self.__window.click(coords)
        self.__window.move_mouse()
        self.__menu_parser.wait_until_loaded()

        self.solve(False, level_str)

    def solve_set(self, save, set_str):
        self.__transition_to_level_select(save)
        
        levels = self.__menu_parser.parse_levels()
        if set_str not in ['1', '2', '3', '4', '5', '6']:
            raise RuntimeError('Set must be between 1-6 (inclusive)')
        
        level = set_str+'-1'
        self.__window.click(levels[level])
        self.__window.move_mouse()
        self.__menu_parser.wait_until_loaded()
        self.solve(True, level)

    def solve_game(self, save):
        self.__transition_to_level_select(save)
        
        for world in ['1', '2', '3', '4', '5', '6']:
            self.solve_set(world)
            self.__menu_parser.wait_until_loaded()
     
    def back(self):
        self.__window.press_key('esc')
        self.__menu_parser.wait_until_loaded()
        
    def close_game(self):
        self.__window.close()
        
    def load_custom_level(self, level_path):
        self.__transition_to_main_menu()
        
        with open(level_path, 'r') as file:
            level = file.read()
            
        pyperclip.copy(level)
        
        user_levels = self.__menu_parser.parse_user_levels()
        self.__window.click(user_levels)
        self.__window.move_mouse()
        self.__menu_parser.wait_until_loaded()
