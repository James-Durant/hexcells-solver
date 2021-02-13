import numpy as np
import cv2, pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from grid import Grid, Cell

class Parser:
    __hex_mask_path = '../resources/hex_mask.png'
    __hex_threshold = 0.05
    __counter_mask_path = '../resources/counter_mask.png'
    __counter_threshold = 0.01
    __angles = np.asarray([-60, 0, 60])
    
    def __init__(self, window):
        self.__window = window
        self.__hex_contour, self.__counter_contour = Parser.__load_masks()
  
    @property 
    def window(self):
        return self.__window
    
    @staticmethod 
    def __load_masks():
        hex_image = cv2.imread(Parser.__hex_mask_path)
        hex_mask  = cv2.inRange(hex_image, Cell.ORANGE , Cell.ORANGE)
        hex_contour, _ = cv2.findContours(hex_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter_image = cv2.imread(Parser.__counter_mask_path)
        counter_mask  = cv2.inRange(counter_image, Cell.BLUE, Cell.BLUE)
        counter_contour, _ = cv2.findContours(counter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return hex_contour[0], counter_contour[0]
        
    def parse_grid(self):
        image = self.__window.screenshot()

        blue_cells   = self.__parse_cells(image, Cell.BLUE)
        black_cells  = self.__parse_cells(image, Cell.BLACK)
        orange_cells = self.__parse_cells(image, Cell.ORANGE)

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

        cols = int(round((x_max - x_min) / x_spacing) + 1)
        rows = int(round((y_max - y_min) / y_spacing) + 1)
        
        grid = [[None]*cols for _ in range(rows)]

        for cell in cells:
            x, y = cell.image_coords
            col = int(round((x - x_min) / x_spacing))
            row = int(round((y - y_min) / y_spacing))
            cell.grid_coords = (row, col)
            grid[row][col] = cell

        _, remaining = self.__parse_counters(image)
        
        scene = Grid(grid, cells, remaining)
        self.__parse_columns(image, scene)
        
        return scene

    def __parse_cells(self, image, cell_colour):
        mask = cv2.inRange(image, cell_colour, cell_colour)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for contour in contours:
            if cv2.contourArea(contour) > 3000 and cv2.matchShapes(contour, self.__hex_contour, 1, 0) < Parser.__hex_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                
                cropped = image[y+10: y+h-10, x+12: x+w-12]
                digit = self.__parse_cell_digit(cropped, cell_colour)
                
                centre = (x + w//2, y + h//2)
                cell = Cell(centre, w, h, cell_colour, digit) 
                cells.append(cell)

        return cells

    def __parse_cell_digit(self, image, cell_colour):
        if cell_colour == Cell.ORANGE:
            return None
        elif cell_colour == Cell.BLACK:
            lang = 'eng'
            config = '--psm 8 -c tessedit_char_whitelist=0123456?{}-'
        elif cell_colour == Cell.BLUE:
            lang = 'eng'
            config = '--psm 6 -c tessedit_char_whitelist=0123456789'
        else:
            raise RuntimeError('invalid cell colour found')

        thresh = np.where(image == cell_colour, 255, 0).astype(np.uint8)
        
        if 0 not in thresh:
            return None
        else:            
            digit_str = pytesseract.image_to_string(thresh, lang=lang, config=config)
            digit = digit_str.split('\n')[0]
            return None if digit == '\x0c' else digit

    def __parse_counters(self, image):
        mask = cv2.inRange(image, Cell.BLUE, Cell.BLUE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lang = 'counter'
        config = '--psm 6 -c tessedit_char_whitelist=0123456789'
        
        values = []
        for contour in contours:
            if cv2.matchShapes(contour, self.__counter_contour, 1, 0) < Parser.__counter_threshold:
                x,y,w,h = cv2.boundingRect(contour)
                y = round(y+h*0.35)
                h = round(h*0.65)

                cropped = image[y: y+h, x: x+w]
                thresh  = np.where(cropped == Cell.BLUE, 255, 0).astype(np.uint8)

                digit_str = pytesseract.image_to_string(thresh, lang=lang, config=config)
                digit = digit_str.split('\n')[0]
                values.append(digit)
        try:
            return int(values[0]), int(values[1])
        except:
            raise RuntimeError('mistakes and/or remaining OCR failed')

    def __parse_columns(self, image, grid):
        _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = [] 
        for contour in contours:
            if cv2.matchShapes(contour, self.__hex_contour, 1, 0) > Parser.__hex_threshold:
                x, y, w, h = cv2.boundingRect(contour) 
                rects += [(x,y,w,h), (x,y,w,h)]
        
        bounding_boxes, _ = cv2.groupRectangles(rects, 1, 1.5)
        for x, y, w, h in bounding_boxes:
            box_coords = np.asarray([x + w//2, y + h//2])
            nearest_cell = grid.nearest_cell(box_coords)
            nearest_coords = nearest_cell.image_coords
                
            delta_x = box_coords[0] - nearest_coords[0]
            delta_y = nearest_coords[1] - box_coords[1]
            theta = 90 - (180 / np.pi * np.arctan2(delta_y, delta_x))
            angle = Parser.__angles[np.argmin(np.abs(Parser.__angles-theta))]
            
            cropped = image[y-10: y+h+10, x-15: x+w+15]
            
            cv2.imshow('test', cropped)
            cv2.waitKey(0)
            
            centre  = (w//2 + 15, h//2 + 10)
            rot_mat = cv2.getRotationMatrix2D(centre, angle, 1.0)
            rotated = cv2.warpAffine(cropped, rot_mat, cropped.shape[1::-1], 
                                     flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
            
            _, thresh = cv2.threshold(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
       
            lang = 'eng'
            config = '--psm 6 -c tessedit_char_whitelist=0123456789{}-'
        
            digit_str = pytesseract.image_to_string(thresh, lang=lang, config=config)
            digit = digit_str.split('\n')[0]
            
            row, col = nearest_cell.grid_coords
            grid.add_constraint(row, col, digit, angle)
            
    def parse_clicked_cells(self, clicked_cells):
        image = self.__window.screenshot()
        for cell in clicked_cells:
            self.__parse_clicked_cell(image, cell)
            
        _, remaining = self.parse_counters(image)
        return remaining

    def __parse_clicked_cell(self, image, cell):
        cx, cy, w, h = *cell.image_coords, self.__hex_width, self.__hex_height

        x1, x2  = cx-w//2, cx+w//2
        y1, y2  = cy-h//2, cy+h//2
        cropped = image[y1+10: y2-10, x1+10: x2-10]
        
        if np.count_nonzero(cropped == Cell.BLACK) > 10:
            cell.colour = Cell.BLACK
        elif np.count_nonzero(cropped == Cell.BLUE) > 10:
            cell.colour = Cell.BLUE
        else:
            raise RuntimeError('cell must be blue or black after click')

        cell.digit = Parser.__parse_cell_digit(cropped, cell.colour)