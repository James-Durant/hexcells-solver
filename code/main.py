import cv2, pyautogui, win32gui, win32com.client
import numpy as np

from parse import Parser
from solve import Solver
from training import make_counter_dataset, make_blue_digit_dataset, make_black_digit_dataset

class Window:
    def __init__(self, title):
        self.__title = title
        self.__hwnd  = win32gui.FindWindow(None, title)
        if not self.__hwnd:
            raise RuntimeError('Hexcells window not found')
        
    def __to_foreground(self):
        shell = win32com.client.Dispatch('WScript.Shell')
        shell.SendKeys('%')
        win32gui.SetForegroundWindow(self.__hwnd)
    
    def __get_position(self):
        x1, y1, x2, y2 = win32gui.GetClientRect(self.__hwnd)
        x1, y1 = win32gui.ClientToScreen(self.__hwnd, (x1, y1))
        x2, y2 = win32gui.ClientToScreen(self.__hwnd, (x2 - x1, y2 - y1))
        return x1, y1, x2, y2
    
    def screenshot(self):
        self.__to_foreground()
        x1, y1, x2, y2 = self.__get_position()
        image = pyautogui.screenshot(region=(x1, y1, x2, y2))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    def click_cell(self, cell, button):
        x1, y1, _, _ = self.__get_position()
        x2, y2       = cell.image_coords
        pyautogui.click(x=x1+x2, y=y1+y2, button=button)
    
def get_window():
    try:
        window = Window('Hexcells')
    except RuntimeError:
        try:
            window = Window('Hexcells Plus')
        except RuntimeError:
            window = Window('Hexcells Infinite')
    return window

if __name__ == "__main__":
    window = get_window()
    parser = Parser(window)
    
    Solver.solve(parser)

    #save_path = '../resources/training/counter'
    #make_counter_dataset(parser, save_path, remaining=478)
    
    #save_path = '../resources/training/blue_digits'
    #make_blue_digit_dataset(parser, save_path)
    
    #save_path = '../resources/training/black_digits'
    #make_black_digit_dataset(parser, save_path)