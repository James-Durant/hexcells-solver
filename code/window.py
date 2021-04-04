import pyperclip, cv2, pyautogui
import win32gui, win32con, win32com.client
import numpy as np

class Window:
    def __init__(self, title):
        self.__title = title
        self.__hwnd  = win32gui.FindWindow(None, title)
        if not self.__hwnd:
            raise RuntimeError('Hexcells window not found') 
            
        x1, y1, x2, y2 = self._get_position()
        self.__resolution = '{0}x{1}'.format(x2-x1, y2-y1)
        
    @property
    def resolution(self):
        return self.__resolution
        
    def _to_foreground(self):
        shell = win32com.client.Dispatch('WScript.Shell')
        shell.SendKeys('%')
        win32gui.SetForegroundWindow(self.__hwnd)
        
    def _get_position(self):
        x1, y1, x2, y2 = win32gui.GetClientRect(self.__hwnd)
        x1, y1 = win32gui.ClientToScreen(self.__hwnd, (x1, y1))
        x2, y2 = win32gui.ClientToScreen(self.__hwnd, (x2 - x1, y2 - y1))
        return x1, y1, x2, y2   
    
    def screenshot(self):
        self._to_foreground()
        x1, y1, x2, y2 = self._get_position()
        image = pyautogui.screenshot(region=(x1, y1, x2, y2))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    def move_mouse(self):
        x1, y1, _, _ = self._get_position()
        pyautogui.moveTo(x1, y1)
    
    def click(self, coords, button='left'):
        x, y, _, _ = self._get_position()
        x += coords[0]
        y += coords[1]
        pyautogui.click(x=x, y=y, button=button)
    
    def close(self):
        win32gui.PostMessage(self.__hwnd, win32con.WM_CLOSE, 0, 0)
    
class ConfigWindow(Window):
    def __init__(self):
        super().__init__('Hexcells Infinite Configuration')
    
    def load_game(self):
        self._to_foreground()
        pyautogui.press('tab')
        pyautogui.press('up') 
        pyautogui.press('enter')
        
    def reset_resolution(self):
        self._to_foreground()
        pyautogui.press('tab')
        pyautogui.press('down', presses=14) 
        pyautogui.press('enter')

class GameWindow(Window): 
    def click_cell(self, cell, button):
        x1, y1, _, _ = self._get_position()
        x2, y2 = cell.image_coords
        pyautogui.click(x=x1+x2, y=y1+y2, button=button)
        
    def load_custom_level(self, level_path):
        with open(level_path, 'r') as file:
            level = file.read()
        pyperclip.copy(level)
        
        x1, y1, x2, y2 = self._get_position()
        x = round((x1+x2)*0.92)
        y = round((y1+y2)*0.97)
        
        self._to_foreground()
        for i in range(5):
            pyautogui.click(x=x, y=y, button='left')
    
def get_window():
    try:
        window = GameWindow('Hexcells')
    except RuntimeError:
        try:
            window = GameWindow('Hexcells Plus')
        except RuntimeError:
            try:
                window = GameWindow('Hexcells Infinite')
            except RuntimeError:
                raise RuntimeError('Hexcells client not found')
    return window