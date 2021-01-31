class Constraint:
    def __init__(self, size, hint, members):
        self.__size = size
        self.__hint = hint
        self.__members = members
        
    @property 
    def size(self):
        return self.__size 
    
    @property 
    def hint(self):
        return self.__hint 
    
    @property 
    def members(self):
        return self.__members

class Grid:
    __FLOWER_DELTAS = [( 0, -2), ( 0, -4), ( 1, -3),
                       ( 1, -1), ( 2, -2), ( 2,  0),
                       ( 1,  1), ( 2,  2), ( 1,  3),
                       ( 0,  2), ( 0,  4), (-1,  3),
                       (-1,  1), (-2,  2), (-2,  0),
                       (-1, -1), (-2, -2), (-1, -3)]
    
    __DIRECT_DELTAS = __FLOWER_DELTAS[::3]
    
    def __init__(self, grid, remaining):
        self.__grid = grid
        self.__rows = len(grid)
        self.__cols = len(grid[0])
        self.__remaining = remaining
        self.__constraints = []
    
    @property
    def rows(self):
        return self.__rows
    
    @property
    def cols(self):
        return self.__cols
    
    @property
    def remaining(self):
        return self.__remaining
    
    @remaining.setter
    def remaining(self, remaining):
        if remaining >= 0:
            self.__remaining = remaining
        else:
            raise RuntimeError('number remaining must be greater than 0')
     
    @property    
    def constraints(self):
        return self.__constraints
    
    def add_constraint(self, row, col, digit, angle):
        if digit is None:
            return
        
        if len(digit) == 1:
            hint = 'normal'
        elif len(digit) == 3:
            if digit[0] == '{' and digit[-1] == '}':
                hint = 'consecutive'
                digit = digit[1:-1]
                
            elif digit[0] == '-' and digit[-1] == '-':
                hint = 'non-consecutive'
                digit = digit[1:-1]
            
            else:
                print(digit)
                raise RuntimeError('consecutive/non-consecutive grid hint parsed incorrectly')
    
        else:
            raise RuntimeError('grid constraint parsed incorrectly')    
        
        try:
            size = int(digit)
        except ValueError:
            raise RuntimeError('grid constraint parsed incorrectly')    
        
        if angle == 0:
            cells = [self[(i,col)] for i in range(row, self.__rows+1) if self[(i,col)] != None]
            
        elif angle == -60:
            cells = []
            while 0 <= row < self.__rows and 0<= col < self.__cols:
                if self[row,col] != None:
                    cells.append(self[row,col])
                row += 1
                col += 1
            
        elif angle == 60:
            cells = []
            while 0 <= row < self.__rows and 0<= col < self.__cols:
                if self[row,col] != None:
                    cells.append(self[row,col])
                row += 1
                col -= 1
            
        else:
            raise RuntimeError('invalid grid constraint angle')

        self.__constraints.append(Constraint(size, hint, cells))
        
    def __cells(self):
        cells = []
        for row in range(self.__rows):
            for col in range(self.__cols):
                cell = self[(row, col)]
                if cell != None:
                    cells.append(cell)
        return cells  
    
    def known_cells(self):
        return [cell for cell in self.__cells() if cell.colour != Cell.ORANGE]   

    def unknown_cells(self):
        return [cell for cell in self.__cells() if cell.colour == Cell.ORANGE]  
    
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
    BLACK  = (62, 62, 62)
    ORANGE = (41, 177, 255) 
    ORANGE_OLD = (41, 175, 255) 
    
    def __init__(self, image_coords, width, height, colour, digit=None):
        self.__image_coords = image_coords
        self.__grid_coords  = None
        self.__width  = width
        self.__height = height
        self.__hint = 'normal'
        self.colour = colour
        self.digit  = digit

    @property
    def image_coords(self):
        return self.__image_coords
    
    @property 
    def width(self):
        return self.__width
    
    @property 
    def height(self):
        return self.__height
    
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
        if colour not in [Cell.BLUE, Cell.BLACK, Cell.ORANGE, Cell.ORANGE_OLD]:
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
            if digit == '?':
                self.__digit = '?'
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
            
        if self.__hint == 'consecutive':
            digit = '{' + str(self.__digit) + '}'
        elif self.__hint == 'non-consecutive':
            digit = '-' + str(self.__digit) + '-'
        elif self.__hint == 'normal' or self.__digit == '?':
            if self.__digit is None:
                digit = '   '
            else:
                digit = ' ' + str(self.__digit) + ' '
            
        return colour + digit
