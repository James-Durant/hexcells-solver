class Grid:
    __FLOWER_DELTAS = [( 0, -2), ( 0, -4), ( 1, -3),
                       ( 1, -1), ( 2, -2), ( 2,  0),
                       ( 1,  1), ( 2,  2), ( 1,  3),
                       ( 0,  2), ( 0,  4), (-1,  3),
                       (-1,  1), (-2,  2), (-2,  0),
                       (-1, -1), (-2, -2), (-1, -3)]
    
    __DIRECT_DELTAS = __FLOWER_DELTAS[::3]
    
    def __init__(self, grid):
        self.__grid = grid
        self.__rows = len(grid)
        self.__cols = len(grid[0])
      
    @property
    def rows(self):
        return self.__rows
    
    @property
    def cols(self):
        return self.__cols
    
    def known_cells(self):
        known = []
        for row in range(self.__rows):
            for col in range(self.__cols):
                cell = self[(row, col)]
                if cell != None and (cell.colour == Cell.BLACK or cell.colour == Cell.BLUE):
                    known.append(cell)
        return known
    
    def unknown_cells(self):
        unknown = []
        for row in range(self.__rows):
            for col in range(self.__cols):
                cell = self[(row, col)]
                if cell != None and cell.colour == Cell.ORANGE:
                    unknown.append(cell)
        return unknown        
    
    def neighbours(self, cell):
        deltas = []
        if cell.colour == Cell.BLACK and cell.digit != '?':
            deltas = Grid.__DIRECT_DELTAS
        elif cell.colour == Cell.BLUE and cell.digit != None:
            deltas = Grid.__FLOWER_DELTAS
        return self.__find_neighbours(cell, deltas)
        
    def __find_neighbours(self, cell, deltas):
        row, col = cell.grid_coords
        return [self[row+d_row, col+d_col] for d_col, d_row in deltas]
    
    def __getitem__(self, key):
        row, col = key
        if 0 <= row < self.__rows and 0 <= col < self.__cols:
            return self.__grid[row][col]
        return None
    
    def __str__(self):
        return_str = ''
        for row in range(self.__rows):
            for col in range(self.__cols):
                cell = self[(row, col)]
                if cell is None:
                    cell_str = '    ' 
                else:
                    cell_str = str(cell)
                return_str = return_str + cell_str + '|'
                
            return_str = return_str[:-1] + '\n'
            if row != self.__rows-1:
                return_str = return_str + '_'*self.__cols*5 + '\n'
                
        return return_str

class Cell:
    BLUE   = (235, 164, 5)
    BLACK  = (62,  62,  62)
    ORANGE = (41,  177, 255)
    
    def __init__(self, image_coords, colour, digit=None):
        self.__image_coords = image_coords
        self.__grid_coords  = None
        self.__hint = 'normal'
        self.colour = colour
        self.digit  = digit

    @property
    def image_coords(self):
        return self.__image_coords
    
    @property
    def grid_coords(self):
        return self.__grid_coords
    
    @grid_coords.setter
    def grid_coords(self, grid_coords):
        self.__grid_coords = grid_coords
    
    @property
    def colour(self):
        return self.__colour
    
    @colour.setter
    def colour(self, colour):
        if colour not in [Cell.BLUE, Cell.BLACK, Cell.ORANGE]:
            raise RuntimeError('invalid cell colour')
        else:
            self.__colour = colour
    
    @property
    def digit(self):
        return self.__digit
    
    @digit.setter
    def digit(self, digit):
        if self.__colour == Cell.BLUE:
            if digit == None:
                self.__digit = None
            else:
                if '-' in digit:
                    raise RuntimeError('blue cell digit cannot be -')
                    
                if '{' in digit or '}' in digit:
                    raise RuntimeError('blue cell digit cannot be { or }')    
                
                if '?' in digit:
                    raise RuntimeError('blue cell digit cannot be ?')    
                
                try:
                    self.__digit = int(digit)
                except ValueError:
                    raise RuntimeError('blue cell digit parsed incorrectly')       
            
        elif self.__colour == Cell.BLACK:
            if digit == None:
                raise RuntimeError('OCR missed black cell digit')
            elif digit == '?':
                self.__digit = None
            else:
                if digit[0] == '{' and digit[-1] == '}':
                    if len(digit) == 2:
                        raise RuntimeError('consecutive black cell missing digit')
                    self.__hint = 'consecutive'
                    digit = digit[1:-1]
                    
                elif digit[0] == '-' and digit[-1] == '-':
                    if len(digit) == 2:
                        raise RuntimeError('non-consecutive black cell missing digit')
                    self.__hint = 'non-consecutive'
                    digit = digit[1:-1]
                    
                try:
                    self.__digit = int(digit)
                except ValueError:
                    raise RuntimeError('black cell digit parsed incorrectly')       
                
        elif self.__colour == Cell.ORANGE:
            if digit != None:
                raise RuntimeError('orange cells cannot have digits')
            else:
                self.__digit = None
        
    @property
    def hint(self):
        return self.__hint
    
    def __str__(self):
        if self.__colour == Cell.BLUE:
            colour = 'X'
        elif self.__colour == Cell.BLACK:
            colour = 'O'
        elif self.__colour == Cell.ORANGE:
            colour = '?'
            
        if self.__hint == "consecutive":
            digit = "{" + str(self.__digit) + "}"
        elif self.__hint == "non-consecutive":
            digit = "-" + str(self.__digit) + "-"
        elif self.__hint == "normal":
            if self.__digit is None:
                digit = '   '
            else:
                digit = ' ' + str(self.__digit) + ' '
            
        return colour + digit
