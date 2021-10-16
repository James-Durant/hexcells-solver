import cv2, pyautogui, win32gui, win32con, win32com.client
import numpy as np

class Window:
    def __init__(self, title):
        self.__title = title
        self.__hwnd  = win32gui.FindWindow(None, title)
        if not self.__hwnd:
            raise RuntimeError('Hexcells window not found') 
            
        x1, y1, x2, y2 = self._get_position()
      
    @property
    def title(self):
        return self.__title
      
    def _get_position(self):
        x1, y1, x2, y2 = win32gui.GetClientRect(self.__hwnd)
        x1, y1 = win32gui.ClientToScreen(self.__hwnd, (x1, y1))
        x2, y2 = win32gui.ClientToScreen(self.__hwnd, (x2 - x1, y2 - y1))
        return x1, y1, x2, y2   
    
    def to_foreground(self):
        shell = win32com.client.Dispatch('WScript.Shell')
        shell.SendKeys('%')
        win32gui.SetForegroundWindow(self.__hwnd)
    
    def screenshot(self):
        self.to_foreground()
        x1, y1, x2, y2 = self._get_position()
        image = pyautogui.screenshot(region=(x1, y1, x2, y2))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    def press_key(self, key):
        pyautogui.press(key)
        
    def click(self, coords, button='left'):
        x, y, _, _ = self._get_position()
        x += coords[0]
        y += coords[1]
        pyautogui.click(x=x, y=y, button=button)
    
    def close(self):
        win32gui.PostMessage(self.__hwnd, win32con.WM_CLOSE, 0, 0)

class GameWindow(Window): 
    def click_cell(self, cell, button):
        x1, y1, _, _ = self._get_position()
        x2, y2 = cell.image_coords
        pyautogui.click(x=x1+x2, y=y1+y2, button=button)
        
    def move_mouse(self):
        x, y, w, h = self._get_position()
        pyautogui.moveTo(x+w-1, y+h-1)  

class ConfigWindow(Window):
    def __init__(self):
        super().__init__('Hexcells Infinite Configuration')
    
    def start_game(self):
        self.to_foreground()
        pyautogui.press('tab')
        pyautogui.press('up') 
        pyautogui.press('enter')
        
    def reset_resolution(self):
        self.to_foreground()
        pyautogui.press('tab')
        pyautogui.press('down', presses=14) 
        pyautogui.press('enter')
    
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
