import cv2
import win32gui
import win32con
import pyautogui
import win32com.client
import numpy as np


class WindowNotFoundError(Exception):
    """Custom exception for when a Hexcells window cannot be found."""
    pass


class Window:
    """Encapsulates a Hexcells game window."""

    def __init__(self, title):
        """Finds a Hexcells game window if one is currently running.

        Args:
            title (str): the game to find the window for.
        """
        # Try to find the window with the given title.
        self.__title = title
        self.__hwnd = win32gui.FindWindow(None, title)
        if not self.__hwnd:
            raise RuntimeError('Hexcells window not found')

    @property
    def title(self):
        """
        Returns:
            str: title of the window.
        """
        return self.__title

    def __get_position(self):
        """Get the position of the game window.

        Returns:
            tuple: monitor coordinates of the window.
        """
        # Get the coordinates of the window and adjust to remove the window's border.
        x1, y1, x2, y2 = win32gui.GetClientRect(self.__hwnd)
        x1, y1 = win32gui.ClientToScreen(self.__hwnd, (x1, y1))
        x2, y2 = win32gui.ClientToScreen(self.__hwnd, (x2 - x1, y2 - y1))
        return x1, y1, x2, y2

    def __to_foreground(self):
        """Bring the window to the foreground."""
        shell = win32com.client.Dispatch('WScript.Shell')
        shell.SendKeys('%')
        win32gui.SetForegroundWindow(self.__hwnd)

    def screenshot(self):
        """
        Returns:
            numpy.ndarray: a screenshot of the active game window.
        """
        # Bring the window to the foreground to take the screenshot.
        self.__to_foreground()
        # Take a screenshot and crop it to the region of the window.
        x1, y1, x2, y2 = self.__get_position()
        image = pyautogui.screenshot(region=(x1, y1, x2, y2))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def press_key(key):
        """Press a given key once.

        Args:
            key (str): key to press.
        """
        pyautogui.press(key)

    @staticmethod
    def paste():
        """Paste the contents of the clipboard in the window."""
        pyautogui.hotkey('ctrl', 'v')

    def click(self, coords, button='left', move_mouse=True):
        """Perform a mouse click at given coordinates.

        Args:
            coords (tuple): coordinates of the object to click.
            button (str, optional): either left or right.
            move_mouse (bool, optional): whether to move the mouse after clicking.
        """
        # Transform the window coordinates to monitor coordinates.
        x, y, _, _ = self.__get_position()
        pyautogui.click(x=x+coords[0], y=y+coords[1], button=button)

        # Move the mouse out of the way if requested.
        if move_mouse:
            self.move_mouse()

    def click_cell(self, cell, button):
        """Click on a cell with a chosen button action.

        Args:
            cell (grid.Cell): cell to click.
            button (str): either left or right.
        """
        # Convert the cell's window coordinates to monitor coordinates.
        x1, y1, _, _ = self.__get_position()
        x2, y2 = cell.image_coords
        pyautogui.click(x=x1 + x2, y=y1 + y2, button=button)

    def move_mouse(self):
        """Move the mouse to the bottom right corner of the window."""
        x, y, w, h = self.__get_position()
        pyautogui.moveTo(x + w - 1, y + h - 1)

    def close(self):
        """Close the window."""
        win32gui.PostMessage(self.__hwnd, win32con.WM_CLOSE, 0, 0)


def get_window():
    """Try to capture an active Hexcells game window.

    Returns:
        window.Window: the active game window.
    """
    # Try to find a window for each game in the Hexcells series.
    try:
        return Window('Hexcells')
    except RuntimeError:
        try:
            return Window('Hexcells Plus')
        except RuntimeError:
            try:
                return Window('Hexcells Infinite')
            except RuntimeError:
                raise WindowNotFoundError('Hexcells window not found')
