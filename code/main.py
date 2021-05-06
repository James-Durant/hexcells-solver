import numpy as np
import cv2, time

from window import get_window
from parse import Parser
from solve import Solver

class Navigator:
    def __init__(self):
        self.__window = get_window()
        image = self.__window.screenshot()
        
        _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        areas = [cv2.contourArea(contour) for contour in contours]
        median_area = np.median(areas)
        
        coords = []
        for i in range(len(contours)):
            if 0.98*median_area < areas[i] < 1.02*median_area:
                x, y, w, h = cv2.boundingRect(contours[i])
                coords.append((x+w//2, y+h//2))

        coords.sort(reverse=True)

        # Need to do image hashing
        labels = ['3-3', '3-2', '2-3', '2-2', '3-4', '3-1',
                  '2-4', '2-1', '3-5', '3-6', '2-5', '2-6',
                  '4-3', '4-2', '1-3', '1-2', '4-4', '4-1',
                  '1-4', '1-1', '4-5', '4-6', '1-5', '1-6',
                  '5-3', '5-2', '6-3', '6-2', '5-4', '5-1',
                  '6-4', '6-1', '5-5', '5-6', '6-5', '6-6']
        
        self.__levels = {label: (x,y) for label, (x,y) in list(zip(labels, coords))}

    def __solve(self, continuous):
        parser = Parser(self.__window)
        solver = Solver(parser)
        solver.solve(continuous=continuous)

    def solve_level(self, level):
        try:
            coords = self.__levels[level]
        except KeyError:
            raise RuntimeError('invalid level given')
            
        self.__window.click(coords)
        time.sleep(1.5)
        self.__solve(continuous=False)

    def solve_world(self, world):
        if world not in ['1', '2', '3', '4', '5', '6']:
            raise RuntimeError('world must be between 1-6 (inclusive)')
        
        self.__window.click(self.__levels[world+'-1'])
        time.sleep(1.5)
        self.__solve(continuous=True)

    def solve_game(self):
        for world in ['1', '2', '3', '4', '5', '6']:
            self.__window.click(self.__levels[world+'-6'])
            time.sleep(1.5)
            self.__solve(continuous=True)
            time.sleep(2)
        

if __name__ == '__main__':
    menu = Navigator()
    #menu.solve_level('1-5')
    #menu.solve_world('1')
    menu.solve_game()