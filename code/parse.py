from grid import Grid, Cell

import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class Parser:
    def __init__(self):
        self.__hex_contour = Parser.__get_hex_contour()
      
    @staticmethod
    def __get_hex_contour(filename='./resources/hex_mask.png'):
        image = cv2.imread(filename)
        mask  = cv2.inRange(image, Cell.ORANGE, Cell.ORANGE)
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0] 
    
    def parse_clicked_cell(self, image, cell):
        cx, cy, w, h = *cell.image_coords, round(self.__avg_hex_width), round(self.__avg_hex_height)
        
        x1, x2  = cx-w//2, cx+w//2
        y1, y2  = cy-h//2, cy+h//2
        cropped = image[y1+10: y2-10, x1+10: x2-10]
        
        if tuple(cropped[5,5]) == Cell.BLACK:
            cell.colour = Cell.BLACK 
        elif tuple(cropped[5,5]) == Cell.BLUE: 
            cell.colour = Cell.BLUE 
        else:
            #cv2.imshow('Cell', cropped)
            #cv2.waitKey(0)
            raise RuntimeError('cell must be blue or black after click')
            
        _, thresh = cv2.threshold(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY_INV)
        digit_str = pytesseract.image_to_string(thresh, lang='eng', config='--psm 10 -c tessedit_char_whitelist=?{}-0123456789')
        digit = digit_str.split('\n')[0]
        if digit == '\x0c':
            digit = None
        cell.digit = digit    
    
    def parse_grid(self, image):
        self.__hex_widths, self.__hex_heights = 0, 0
        self.__x_min, self.__x_max = float("inf"), -float("inf")
        self.__y_min, self.__y_max = float("inf"), -float("inf")
        
        blue_cells   = self.__parse_cell_colour(image, Cell.BLUE)
        black_cells  = self.__parse_cell_colour(image, Cell.BLACK)
        orange_cells = self.__parse_cell_colour(image, Cell.ORANGE)
        
        cells = blue_cells + black_cells + orange_cells

        self.__avg_hex_width  = self.__hex_widths  / len(cells)
        self.__avg_hex_height = self.__hex_heights / len(cells)
        
        x_spacing = self.__avg_hex_width*1.085 #88
        y_spacing = self.__avg_hex_height*0.72 #50   
        
        cols = round((self.__x_max - self.__x_min) / x_spacing) + 1
        rows = round((self.__y_max - self.__y_min) / y_spacing) + 1
        grid = [[None]*cols for _ in range(rows)]

        for cell in cells:
            x, y = cell.image_coords
            col = round((x - self.__x_min) / x_spacing)
            row = round((y - self.__y_min) / y_spacing)
            cell.grid_coords = (row, col)
            grid[row][col] = cell

        return Grid(grid)
    
        #cv2.imshow('Parsed Grid', image)
        #cv2.waitKey(0)
    
    def __parse_cell_colour(self, image, cell_colour, match_threshold=0.05):
        mask     = cv2.inRange(image, cell_colour, cell_colour)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        cells = []
        for contour in contours:
            if cv2.matchShapes(contour, self.__hex_contour, 1, 0) < match_threshold:
                x,y,w,h = cv2.boundingRect(contour)
            
                digit = None
                if cell_colour != Cell.ORANGE:
                    cropped   = image[y+10: y+h-10, x+10: x+w-10]
                    _, thresh = cv2.threshold(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY_INV)
                    digit_str = pytesseract.image_to_string(thresh, lang='eng', config='--psm 10 -c tessedit_char_whitelist=?{}-0123456789')
                    digit = digit_str.split('\n')[0]
                    if digit == '\x0c':
                        digit = None
                    #else:
                    #    cv2.rectangle(image, (x+10,y+10), (x+w-10,y+h-10), (0,0,255), 1)
        
                cx, cy = x + w//2, y + h//2
                
                self.__hex_widths  += w
                self.__hex_heights += h
                self.__x_min = min(self.__x_min, cx)
                self.__x_max = max(self.__x_max, cx)
                self.__y_min = min(self.__y_min, cy)
                self.__y_max = max(self.__y_max, cy)
                
                cell = Cell((cx, cy), cell_colour, digit)
                cells.append(cell)
                #cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)
                #cv2.putText(image, str((cx, cy)), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            
        return cells  
    