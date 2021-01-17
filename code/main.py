import numpy as np
import cv2, pyautogui
import win32gui, win32com.client

from parse import Parser
from solve import Solver

class Window:
    def __init__(self, title):
        self.__title = title
        self.__hwnd  = win32gui.FindWindow(None, title)
        if not self.__hwnd:
            raise RuntimeError('Hexcells window not found')
        self.__parser = Parser()
        
    def __to_foreground(self):
        shell = win32com.client.Dispatch('WScript.Shell')
        shell.SendKeys('%')
        win32gui.SetForegroundWindow(self.__hwnd)
    
    def __get_position(self):
        x1, y1, x2, y2 = win32gui.GetClientRect(self.__hwnd)
        x1, y1 = win32gui.ClientToScreen(self.__hwnd, (x1, y1))
        x2, y2 = win32gui.ClientToScreen(self.__hwnd, (x2 - x1, y2 - y1))
        return x1, y1, x2, y2
    
    def __screenshot(self):
        self.__to_foreground()
        x1, y1, x2, y2 = self.__get_position()
        image = pyautogui.screenshot(region=(x1, y1, x2, y2))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    def parse_grid(self):
        return self.__parser.parse_grid(self.__screenshot())
    
    def parse_clicked_cells(self, clicked_cells):
        image = self.__screenshot()
        for i, cell in enumerate(clicked_cells):
            self.__parser.parse_clicked_cell(image, cell)
    
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
    grid = window.parse_grid()
    Solver.solve(window, grid)
