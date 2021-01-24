import numpy as np
import cv2, pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from grid import Grid, Cell

class Parser:
    counter_threshold = 0.1
    cell_threshold = 0.05
    
    def __init__(self, window):
        self.__window = window
        self.__hex_contour = Parser.__get_hex_contour()
        self.__counter_contour = Parser.__get_counter_contour()

    @property 
    def window(self):
        return self.__window

    @staticmethod
    def __get_hex_contour(filename='../resources/hex_mask.png'):
        image = cv2.imread(filename)
        mask  = cv2.inRange(image, Cell.ORANGE , Cell.ORANGE)
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]

    @staticmethod
    def __get_counter_contour(filename='../resources/counter_mask.png'):
        mask = cv2.inRange(cv2.imread(filename), Cell.BLUE, Cell.BLUE)
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]

    @staticmethod
    def __parse_cell_digit(image, cropped, cell_colour, save_path=None):
        _, thresh = cv2.threshold(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY_INV)
        
        if save_path != None:
            cv2.imwrite(save_path, thresh)
        
        config = '--psm 6 -c tessedit_char_whitelist=0123456789'
        if cell_colour == Cell.BLACK:
            lang = 'black_digits'
            config = '--psm 6 -c tessedit_char_whitelist=0123456?{}-'
        elif cell_colour == Cell.BLUE:
            lang = 'blue_digits'
        else:
            lang = 'eng'
        
        digit_str = pytesseract.image_to_string(thresh, lang=lang, config=config)
        digit = digit_str.split('\n')[0]
        return None if digit == '\x0c' else digit

    def parse_counters(self, image, save_path=None):
        mask     = cv2.inRange(image, Cell.BLUE, Cell.BLUE)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        boxes = []
        for contour in contours:
            if cv2.matchShapes(contour, self.__counter_contour, 1, 0) < Parser.counter_threshold:
                x,y,w,h = cv2.boundingRect(contour)
                y = round(y+h*0.35)
                h = round(h*0.65)
                boxes.append((x,y,w,h))

        if len(boxes) != 2:
            raise RuntimeError('mistakes and/or remaining box not found')
        else:
            x,y,w,h = boxes[1]
            cropped = image[y: y+h, x: x+w]
            _, thresh = cv2.threshold(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY_INV)
            
            if save_path != None:
                cv2.imwrite(save_path, thresh)
            
            digit_str = pytesseract.image_to_string(thresh, lang='counter', config='--psm 6 -c tessedit_char_whitelist=0123456789')
            digit = digit_str.split('\n')[0]
            try:
                return 0, int(digit)
            except ValueError:
                raise RuntimeError('mistakes and/or remaining OCR failed')

    def parse_clicked_cells(self, clicked_cells):
        image = self.__window.screenshot()
        for cell in clicked_cells:
            self.__parse_clicked_cell(image, cell)
            
        _, remaining = self.parse_counters(image)
        return remaining

    def __parse_clicked_cell(self, image, cell):
        cx, cy, w, h = *cell.image_coords, round(self.__hex_width), round(self.__hex_height)

        x1, x2  = cx-w//2, cx+w//2
        y1, y2  = cy-h//2, cy+h//2
        cropped = image[y1+10: y2-10, x1+10: x2-10]
        
        if np.count_nonzero(cropped == Cell.BLACK) >= 10:
            cell.colour = Cell.BLACK
        elif np.count_nonzero(cropped == Cell.BLUE) >= 10:
            cell.colour = Cell.BLUE
        else:
            cv2.imshow('Cell', cropped)
            cv2.waitKey(0)
            raise RuntimeError('cell must be blue or black after click')

        cell.digit = Parser.__parse_cell_digit(image, cropped, cell.colour)

    def parse_grid(self):
        image = self.__window.screenshot()

        blue_cells   = self.parse_cells_of_colour(image, Cell.BLUE)
        black_cells  = self.parse_cells_of_colour(image, Cell.BLACK)
        orange_cells = self.parse_cells_of_colour(image, Cell.ORANGE)

        cells = blue_cells + black_cells + orange_cells

        widths  = [cell.width  for cell in cells]     
        heights = [cell.height for cell in cells]   
        xs = [cell.image_coords[0] for cell in cells]
        ys = [cell.image_coords[1] for cell in cells]
        
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        self.__hex_width  = int(np.median(widths))
        self.__hex_height = int(np.median(heights))

        x_spacing = self.__hex_width*1.085
        y_spacing = self.__hex_height*0.70 

        print((x_max - x_min) / x_spacing)
        print((y_max - y_min) / y_spacing)

        cols = int(round((x_max - x_min) / x_spacing) + 1)
        rows = int(round((y_max - y_min) / y_spacing) + 1)
        
        grid = [[None]*cols for _ in range(rows)]

        for cell in cells:
            x, y = cell.image_coords
            col = int(round((x - x_min) / x_spacing))
            row = int(round((y - y_min) / y_spacing))
            cell.grid_coords = (row, col)
            grid[row][col] = cell

        _, remaining = self.parse_counters(image)
        return Grid(grid, remaining)

    def parse_cells_of_colour(self, image, cell_colour, save_paths=None):
        mask     = cv2.inRange(image, cell_colour, cell_colour)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        cells = []
        for i, contour in enumerate(contours):
            if cv2.matchShapes(contour, self.__hex_contour, 1, 0) < Parser.cell_threshold:
                x,y,w,h = cv2.boundingRect(contour)

                if cell_colour == Cell.ORANGE or cell_colour == Cell.ORANGE_OLD:
                    digit = None
                else:
                    cropped = image[y+10: y+h-10, x+10: x+w-10]
                    path = None if save_paths == None else save_paths[i]
                    digit = Parser.__parse_cell_digit(image, cropped, cell_colour, path)
            
                cx, cy = x + w//2, y + h//2
                cells.append(Cell((cx, cy), w, h, cell_colour, digit))
                #cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)

        return cells