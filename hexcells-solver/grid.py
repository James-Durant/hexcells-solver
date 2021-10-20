import numpy as np

class Constraint:
    def __init__(self, size, hint, angle, members):
        self.__size = size
        self.__hint = hint
        self.__angle = angle
        self.__members = members

    @property
    def size(self):
        return self.__size

    @property
    def hint(self):
        return self.__hint

    @property
    def angle(self):
        return self.__angle

    @property
    def members(self):
        return self.__members

class Grid:
    __DIRECT = [(0,-2), (1,-1), (1,1), (0,2), (-1,1), (-1,-1)]

    __OUTER = [(0,-4), (1,-3), (2,-2), (2,0), (2,2), (1,3), (0,4),
               (-1,3), (-2,2), (-2,0), (-2,-2), (-1,-3)]

    __FLOWER = __DIRECT + __OUTER

    def __init__(self, grid, cells, remaining):
        self.__grid = grid
        self.__rows = len(grid)
        self.__cols = len(grid[0])
        self.__remaining = remaining
        self.__constraints = []

        self.__cells = cells
        self.__cell_image_coords = np.asarray([cell.image_coords for cell in cells])

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

    @property
    def cells(self):
        return self.__cells

    def nearest_cell(self, query_coords):
        nearest = np.argmin(np.sum((self.__cell_image_coords - query_coords)**2, axis=1))
        return self.__cells[nearest]

    def add_constraint(self, row, col, digit, angle):
        if digit[0] == '{' and digit[-1] == '}':
            hint = 'consecutive'
            digit = digit[1:-1]
        elif digit[0] == '-' and digit[-1] == '-':
            hint = 'non-consecutive'
            digit = digit[1]
        else:
            hint = 'normal'

        try:
            size = int(digit)
        except ValueError:
            raise RuntimeError('grid constraint parsed incorrectly')

        if angle == 360:
            angle = 0

        cells = []
        while 0 <= row < self.__rows and 0 <= col < self.__cols:
            cell = self[(row, col)]
            if cell != None:
                cells.append(cell)

            if angle == 0:
                row += 1

            elif angle == 60:
                row += 1
                col -= 1

            elif angle == 90:
                col -= 1

            elif angle == 120:
                row -= 1
                col -= 1

            elif angle == 240:
                row -= 1
                col += 1

            elif angle == 270:
                col += 1

            elif angle == 300:
                row += 1
                col += 1

            else:
                raise RuntimeError('invalid grid constraint angle')

        self.__constraints.append(Constraint(size, hint, angle, cells))

    def known_cells(self):
        return [cell for cell in self.cells if cell.colour != Cell.ORANGE]

    def unknown_cells(self):
        return [cell for cell in self.cells if cell.colour == Cell.ORANGE]

    def neighbours(self, cell):
        deltas = []
        if cell.colour == Cell.BLACK and cell.digit != '?':
            deltas = Grid.__DIRECT
        elif cell.colour == Cell.BLUE and cell.digit != None:
            deltas = Grid.__FLOWER

        return self.__find_neighbours(cell, deltas)

    def direct_neighbours(self, cell):
        return self.__find_neighbours(cell, Grid.__DIRECT)

    def outer_neighbours(self, cell):
        return self.__find_neighbours(cell, Grid.__OUTER)

    def flower_neighbours(self, cell):
        return self.__find_neighbours(cell, Grid.__FLOWER)

    def __find_neighbours(self, cell, deltas):
        row, col = cell.grid_coords
        return [self[row+d_row, col+d_col] for d_col, d_row in deltas]

    def get_column(self, col):
        return [self.__grid[i][col] for i in range(self.__rows) if self.__grid[i][col] is not None]

    def __getitem__(self, key):
        row, col = key
        if 0 <= row < self.__rows and 0 <= col < self.__cols:
            return self.__grid[row][col]
        return None

    #Make this better
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

        return_str = return_str + 'Remaining: ' + str(self.remaining) + '\n'
        return return_str

class Cell:
    BLUE = (235, 164, 5)
    BLACK = (62, 62, 62)
    ORANGE = (41, 177, 255)
    COLOURS = [BLUE, BLACK, ORANGE]
    HINT_TYPES = ['normal', 'consecutive', 'non-consecutive']

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
    def grid_coords(self):
        return self.__grid_coords

    @grid_coords.setter
    def grid_coords(self, grid_coords):
        self.__grid_coords = grid_coords

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def colour(self):
        return self.__colour

    @property
    def hint(self):
        return self.__hint

    @hint.setter
    def hint(self, hint):
        # Validate this
        self.__hint = hint

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
                return

            if digit[0] == '{' and digit[-1] == '}':
                if len(digit) == 2:
                    raise RuntimeError('consecutive blue cell missing digit')
                self.__hint = 'consecutive'
                digit = digit[1:-1]

            elif digit[0] == '-' and digit[-1] == '-':
                if len(digit) == 2:
                    raise RuntimeError('non-consecutive blue cell missing digit')
                self.__hint = 'non-consecutive'
                digit = digit[1:-1]

            try:
                self.__digit = int(digit)
            except ValueError:
                raise RuntimeError('blue cell digit parsed incorrectly')

        elif self.__colour == Cell.BLACK:
            if digit == None:
                raise RuntimeError('OCR missed black cell digit')
            elif digit == '?':
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
