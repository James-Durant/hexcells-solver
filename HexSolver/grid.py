import numpy as np


class Grid:
    """Represents a Hexcells level as a grid of Cell and Constraint objects."""

    # Relative coordinates of cells in a 1-cell neighbourhood.
    INNER = [(0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1)]

    # Relative coordinates of cells in a 2-cell neighbourhood.
    OUTER = [(0, -4), (1, -3), (2, -2), (2, 0), (2, 2), (1, 3), (0, 4),
             (-1, 3), (-2, 2), (-2, 0), (-2, -2), (-1, -3)]

    # All cells contained within a 2-cell radius.
    COMBINED = INNER + OUTER

    # Row and column directions for each grid constraint angle.
    DIRECTIONS = {0: (1, 0),  # Move down a row only.
                  60: (1, -1),  # Move down one row and left one column.
                  90: (0, -1),  # Move left one column only.
                  120: (-1, -1),  # Move up one row and left one column.
                  240: (-1, 1),  # Move up one row and right one column.
                  270: (0, 1),  # Move right one column only.
                  300: (1, 1)}  # Move up one row and right one column.

    def __init__(self, grid, cells, remaining):
        """Encapsulate a given grid of cell objects parsed from a level.

        Args:
            grid (list): layout of the level.
            cells (list): all cells within the level.
            remaining (int): number of blue cells to uncover in the level.
        """
        self.__grid = grid
        self.__rows = len(grid)  # Number of rows.
        self.__cols = len(grid[0])  # Number of columns.
        self.__cells = cells
        self.__remaining = remaining
        self.__constraints = []  # No constraints initially.

        # Get the image coordinates for each cell.
        self.__cell_image_coords = np.asarray([cell.image_coords for cell in cells])

    @property
    def rows(self):
        """
        Returns:
            int: number of rows in the grid.
        """
        return self.__rows

    @property
    def cols(self):
        """
        Returns:
            int: number of columns in the grid.
        """
        return self.__cols

    @property
    def remaining(self):
        """
        Returns:
            int: number of remaining blue cells to uncover.
        """
        return self.__remaining

    @remaining.setter
    def remaining(self, remaining):
        """Set the number of remaining blue cells to uncover.

        Args:
            remaining (int): new number of blue cells to uncover.
        """
        # Check the new value is non-negative.
        if remaining >= 0:
            self.__remaining = remaining
        else:
            raise RuntimeError('Number of remaining blue cells must be greater than 0')

    @property
    def constraints(self):
        """
        Returns:
            list: grid constraints of the level.
        """
        return self.__constraints

    @property
    def cells(self):
        """
        Returns:
            list: all cells within the level.
        """
        return self.__cells

    def nearest_cell(self, query_coords):
        """Get the cell that is closest to the given query coordinates.

        Args:
            query_coords (tuple): image coordinates to find the nearest cell to.

        Returns:
            grid.Cell: nearest cell to the given coordinates.
        """
        # Return the cell with minimum Euclidean distance.
        distances = np.linalg.norm(self.__cell_image_coords - query_coords, axis=1)
        return self.__cells[np.argmin(distances)]

    def add_constraint(self, row, col, parsed, angle):
        """Add a constraint to the grid.

        Args:
            row (int): row that the constraint starts at.
            col (int): column that the constraint starts at.
            parsed (str): parsed representation of the constraint.
            angle (float): orientation of the constraint.

        """
        # Set the hint type of the constraint as consecutive if it was parsed as {n}
        if parsed[0] == '{' and parsed[-1] == '}':
            hint = 'consecutive'
            number = parsed[1:-1]  # Extract the constraint's number.

        # Set the hint type of the constraint as non-consecutive if it was parsed as -n-
        elif parsed[0] == '-' and parsed[-1] == '-':
            hint = 'non-consecutive'
            number = parsed[1:-1]  # Extract the constraint's number.
        else:
            # Otherwise, there is no hint type.
            hint = 'normal'
            number = parsed

        # Try to convert the constraint's number to an integer.
        try:
            size = int(number)

        except ValueError:
            # If this fails, the parsed representation must be incorrect.
            raise RuntimeError('Grid constraint parsed incorrectly')

        # 360 degrees is the same as 0 degrees.
        if angle == 360:
            angle = 0

        # Get the cells that the constraint acts on.
        cells = []
        delta_row, delta_col = Grid.DIRECTIONS[angle]
        while 0 <= row < self.__rows and 0 <= col < self.__cols:
            # Add the cell at the current row and column (if there is one).
            cell = self[(row, col)]
            if cell is not None:
                cells.append(cell)

            # Move to the next cell that the constraint acts on.
            row += delta_row
            col += delta_col

        # Add the constraint to the list of constraints for this level.
        self.__constraints.append(Constraint(size, hint, angle, cells))

    def known_cells(self):
        """
        Returns:
            list: the set of known (i.e., blue or black) cells.
        """
        return [cell for cell in self.cells if cell.colour != Cell.ORANGE]

    def unknown_cells(self):
        """
        Returns:
            list: the set of unknown (i.e., orange) cells.
        """
        return [cell for cell in self.cells if cell.colour == Cell.ORANGE]

    def neighbours(self, cell):
        """Get the neighbours of a given cell.

        Args:
            cell (grid.Cell): cell to get the neighbours of.

        Returns:
            list: neighbours of the given cell.
        """
        # For black cells (ignore cells with ?), return the adjacent cells.
        if cell.colour == Cell.BLACK and cell.number != '?':
            deltas = Grid.INNER

        # For blue cells with a number, return the cells in a 2-cell radius.
        elif cell.colour == Cell.BLUE and cell.number is not None:
            deltas = Grid.COMBINED

        else:
            # Otherwise, there are no neighbours to consider.
            deltas = []

        return self.find_neighbours(cell, deltas)

    def find_neighbours(self, cell, deltas):
        """Get the neighbours of a cell defined by given relative coordinates.

        Args:
            cell (grid.Cell): cell to get the neighbours of.
            deltas (list): positions of surrounding cells to consider.

        Returns:
            list: neighbours of the given cell.
        """
        row, col = cell.grid_coords
        return [self[row + d_row, col + d_col] for d_col, d_row in deltas]

    def get_column(self, col):
        """Get the cells contain in a given column.

        Args:
            col (int): column to get the cells from.

        Returns:
            list: non-empty cells in the given column.
        """
        return [self.__grid[i][col] for i in range(self.__rows) if self.__grid[i][col] is not None]

    def __getitem__(self, key):
        """Get the cell at the given row and column defined by the key.

        Args:
            key (tuple): row and column index for the cell.

        Returns:
            grid.Cell: cell at the position defined by the key.
        """
        # Check that the cell is within the grid.
        row, col = key
        if 0 <= row < self.__rows and 0 <= col < self.__cols:
            return self.__grid[row][col]

        return None

    def __str__(self):
        """Return a string representation of the grid. This is only used for debugging.

        Returns:
            str: string representation of the grid.
        """
        return_str = ''
        # Iterate over each row and column.
        for row in range(self.__rows):
            for col in range(self.__cols):
                cell = self[(row, col)]
                # Check if the cell is blank.
                if cell is None:
                    cell_str = '    '
                else:
                    cell_str = str(cell)

                # Add a seperator after the cell's string representation.
                return_str = return_str + cell_str + '|'

            # Remove the seperator of the last cell and start a new line for the next row.
            return_str = return_str[:-1] + '\n'

            # If this is not the last row, add a horizontal line.
            if row != self.__rows - 1:
                return_str = return_str + '_' * self.__cols * 5 + '\n'

        # Add the number of remaining cells to uncover.
        return return_str + f'Remaining: {self.remaining}\n'


class Cell:
    """Represents an individual cell contained within a Grid object."""

    # The RGB values of the colours used for blue, black and orange cells.
    BLUE = (235, 164, 5)
    BLACK = (62, 62, 62)
    ORANGE = (41, 177, 255)

    # The colours and hint types used in Hexcells.
    COLOURS = [BLUE, BLACK, ORANGE]
    HINT_TYPES = ['normal', 'consecutive', 'non-consecutive']

    def __init__(self, image_coords, width, height, colour, number=None):
        """Store the properties of the cell.

        Args:
            image_coords (tuple): position of the cell in the game window.
            width (int): width of the cell in the game window (in number of pixels).
            height (int): height of the cell in the game window (in number of pixels).
            colour (tuple): cell colour as an RGB tuple.
            number (str, optional): parsed number/hint associated with the cell.
        """
        self.__image_coords = image_coords
        self.__grid_coords = None  # Initially none but is set when parsing.
        self.__width = width
        self.__height = height

        # Use the setters defined below.
        self.colour = colour
        self.number = number

    @property
    def image_coords(self):
        """
        Returns:
            tuple: position of the cell in the game window.
        """
        return self.__image_coords

    @property
    def grid_coords(self):
        """
        Returns:
            tuple: position of the cell in terms of grid coordinates.
        """
        return self.__grid_coords

    @grid_coords.setter
    def grid_coords(self, grid_coords):
        """Set the grid coordinates of the cell.

        Args:
            grid_coords (tuple): row and column values defining the cell's position.
        """
        self.__grid_coords = grid_coords

    @property
    def width(self):
        """
        Returns:
            int: width of the cell in the game window.
        """
        return self.__width

    @property
    def height(self):
        """
        Returns:
            int: height of the cell in the game window.
        """
        return self.__height

    @property
    def hint(self):
        """
        Returns:
            str: hint type associated with the cell.
        """
        return self.__hint

    @hint.setter
    def hint(self, hint):
        """Set the hint type of the cell.

        Args:
            hint (str): hint type to set.
        """
        # Check that the hint type is one of the three valid values.
        if hint in Cell.HINT_TYPES:
            self.__hint = hint
        else:
            raise RuntimeError('Invalid cell hint type given')

    @property
    def colour(self):
        """
        Returns:
            tuple: colour of the cell.
        """
        return self.__colour

    @colour.setter
    def colour(self, colour):
        """Set the colour of the cell.

        Args:
            colour (tuple): cell colour to set.
        """
        # Check that the cell colour is one of the three valid values.
        if colour in Cell.COLOURS:
            self.__colour = colour
        else:
            raise RuntimeError('Invalid cell colour given')

    @property
    def number(self):
        """
        Returns:
            int: number associated with the cell.
        """
        return self.__number

    @number.setter
    def number(self, parsed):
        """Set the number of the cell.

        Args:
            parsed (str): parsed representation of the cell's number/hint.
        """
        # Initially assume that the hint is normal.
        self.__hint = 'normal'

        if parsed is None:
            # Black cells must have a number.
            if self.__colour == Cell.BLACK:
                raise RuntimeError('Parsing missed black cell number')
            else:
                # Blue and orange cells do not have to have a number.
                self.__number = None
                return

        # Only blue and black cells can contain numbers.
        if self.__colour == Cell.ORANGE:
            raise RuntimeError('Orange cell cannot have a number')

        if parsed == '?':
            # If the cell contains "?" and it is black, no further processing required.
            if self.__colour == Cell.BLACK:
                self.__number = '?'
                return
            else:
                # Only black cells can contain "?".
                raise RuntimeError('Only black cells can have "?"')

        # If the parsed representation is of the form {n}, set the hint type as consecutive.
        if parsed[0] == '{' and parsed[-1] == '}':
            self.__hint = 'consecutive'
            number = parsed[1:-1]

        # If the parsed representation is of the form -n-, set the hint type as non-consecutive.
        elif parsed[0] == '-' and parsed[-1] == '-':
            self.__hint = 'non-consecutive'
            number = parsed[1:-1]

        else:
            # Otherwise, the hint type is normal.
            number = parsed

        try:
            # Try to convert the parsed number to an integer.
            self.__number = int(number)

        except ValueError:
            # If this fails, parsing must have failed.
            raise RuntimeError('Cell number parsed incorrectly')

    def __str__(self):
        """
        Returns:
            str: string representation of the cell.
        """
        # Use different symbols for each cell colour.
        if self.__colour == Cell.BLUE:
            colour = 'X'
        elif self.__colour == Cell.BLACK:
            colour = 'O'
        elif self.__colour == Cell.ORANGE:
            colour = '?'
        else:
            # This should never be raised.
            raise RuntimeError('Invalid cell colour found')

        # Concatenate the cell's number and hint type.
        if self.__hint == 'consecutive':
            number = f'{{{self.__number}}}'

        elif self.__hint == 'non-consecutive':
            number = f'-{self.__number}-'

        elif self.__hint == 'normal' or self.__number == '?':
            if self.__number is None:
                number = '   '
            else:
                number = f' {self.__number} '
        else:
            # This should never be raised.
            raise RuntimeError('Invalid cell number/hint found')

        # Concatenate the cell colour and number/hint.
        return colour + number


class Constraint:
    """Represents a grid constraint within a level."""

    # Valid hint types and angles for constraints.
    HINT_TYPES = Cell.HINT_TYPES
    ANGLES = [0, 60, 90, 120, 240, 270, 300, 360]

    def __init__(self, number, hint, angle, members):
        """Set the properties of the grid constraint.

        Args:
            number (int): number associated with the constraint.
            hint (str): hint type associated with the constraint.
            angle (int): orientation of the constraint (in degrees).
            members (list): cells that the constraint acts on.
        """
        # Use the setters defined below.
        self.number = number
        self.hint = hint
        self.angle = angle
        self.__members = members

    @property
    def number(self):
        """
        Returns:
            int: number of blue cells in the row/column/diagonal defined by the constraint.
        """
        return self.__number

    @number.setter
    def number(self, number):
        """Set the number of the constraint.

        Args:
            number (int): number to set for the constraint.
        """
        # Check the number is non-negative.
        if number >= 0:
            self.__number = number
        else:
            raise RuntimeError('Invalid grid constraint number given')

    @property
    def hint(self):
        """
        Returns:
            str: hint type associated with the grid constraint.
        """
        return self.__hint

    @hint.setter
    def hint(self, hint):
        """Set the hint type of the grid constraint.

        Args:
            hint (str): hint type to set.
        """
        # Check that the hint type is one of the three valid values.
        if hint in Constraint.HINT_TYPES:
            self.__hint = hint
        else:
            raise RuntimeError('Invalid grid constraint hint type given')

    @property
    def angle(self):
        """
        Returns:
            int: orientation of the grid constraint.
        """
        return self.__angle

    @angle.setter
    def angle(self, angle):
        """Set the angle for the grid constraint.

        Args:
            angle (int): angle to set.
        """
        # Check that the angle is one of the valid values.
        if angle in Constraint.ANGLES:
            self.__angle = angle
        else:
            raise RuntimeError('Invalid grid constraint angle given')

    @property
    def members(self):
        """
        Returns:
            list: cells that the grid constraint applies to.
        """
        return self.__members
