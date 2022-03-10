import os
import pulp

from grid import Grid, Cell

# File path to the GNU Linear Programming Kit (GLPK) solver.
GLPK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'resources/winglpk-4.65/glpk-4.65/w64/glpsol.exe')

class Solver:
    """Contains the code related to automatically solving Hexcells levels."""

    def __init__(self, parser):
        """Initialises the solver.

        Args:
            parser (parse.GameParser): parser to parse uncovered cell information.
        """
        self.__parser = parser

    def solve(self, game=None, level=None, delay=False):
        """Solves a Hexcells level.

        Args:
            game (str, optional): the game in the series being solved.
            level (str, optional): set and number of the level being solved.
            delay (bool, optional): whether to add a delay after clicking cells.
        """
        # Parse the initial level state.
        grid = self.__parser.parse_grid()

        # Solve the level step-by-step.
        while True:
            # Get the cells to left and right click.
            left_click_cells, right_click_cells = self.solve_single_step(grid, game, level)

            # If the level is solved after clicking the cells, do not parse them.
            # This is to avoid issues with the level completion screen covering information.
            if len(left_click_cells) + len(right_click_cells) - len(self.__unknown) == 0:
                self.__parser.click_cells(left_click_cells, 'left')
                self.__parser.click_cells(right_click_cells, 'right')
                break

            # Otherwise, parse the cells after clicking them.
            # Also, update the number of remaining blue cells to uncover.
            _, remaining = self.__parser.parse_clicked(grid, left_click_cells, right_click_cells, delay)
            grid.remaining = remaining

    def solve_single_step(self, grid, game=None, level=None):
        """Computes the sets of cells to left and right click, given the available information.

        Args:
            grid (grid.Grid): level to solve.
            game (str, optional): the game in the series being solved.
            level (str, optional): set and number of the level being solved.

        Returns:
            tuple: the set of cells to left and right click in the solution.
        """
        # Define the constraints for the ILP problem.
        self.__setup_problem(grid, game, level)

        # Set up a temporary objective function to obtain an initial solution.
        temp = pulp.LpVariable('initial', 0, 1, 'binary')
        self.__problem += temp == 0
        self.__problem.setObjective(temp)
        self.__problem.solve(self.__solver) # Run the solver.

        # Identify the variables that are assigned their maximum (i.e., the size of the equivalence class
        # corresponding to the variable) and those assigned their minimum (i.e., 0).
        # The true and false sets contain the class representatives.
        true_set, false_set = self.__get_true_false_sets()

        # Get the set of cells to left and right click.
        # Keep iterating until there are no cells in the true or false sets.
        left_click_cells, right_click_cells = [], []
        while true_set or false_set:
            # Get the sums of the variables for the classes in true and false sets.
            true_sum = pulp.lpSum(self.__get_var(true) for true in true_set)
            false_sum = pulp.lpSum(self.__get_var(false) for false in false_set)

            # The same ILP is used again but with an updated objective function.
            # The solver will try to find a new solution that varies as much as possible from the initial solution.
            # The variables that were assigned their maximum will be assigned their minimum, and vice versa.
            self.__problem.setObjective(true_sum - false_sum)
            self.__problem.solve(self.__solver)

            # Check if there are changes between the new and previous solutions.
            # If there are no changes, each variable must be assigned to its singular possible value.
            if pulp.value(self.__problem.objective) == sum(len(self.__class_of_rep[rep]) for rep in true_set):
                # For each representative in the true set, add the members of its equivalence class
                # to the cells to be left clicked (i.e., revealed as blue).
                for rep in true_set:
                    left_click_cells.extend(self.__class_of_rep[rep])

                # For each representative in the false set, add the members of its equivalence class
                # to the cells to be right clicked (i.e., revealed as black).
                for rep in false_set:
                    right_click_cells.extend(self.__class_of_rep[rep])

                return left_click_cells, right_click_cells

            # If there are changes, only keep the variables that did not change.
            true_new, false_new = self.__get_true_false_sets()
            true_set = true_set.intersection(true_new)
            false_set = false_set.intersection(false_new)

        # This should never be raised, assuming that the level has a solution.
        raise RuntimeError('Solver failed to solve level.')

    def __get_true_false_sets(self):
        """Get the representatives of the equivalence classes whose corresponding decision variable's
           value is either of minimum size (false set) or maximum size (true set).

        Returns:
            tuple: true and false sets of equivalence class representatives.
        """
        # Iterate over the decision variable for each equivalence class.
        true_set, false_set = set(), set()
        for rep, rep_class in self.__class_of_rep.items():
            # Get the value of the decision variable for the class.
            value = pulp.value(self.__get_var(rep))

            # If the variable is equal to 0, all cells of the class are black.
            if value == 0:
                false_set.add(rep)

            # If the variable is equal to the size of the class, all cells of the class are blue.
            elif value == len(rep_class):
                true_set.add(rep)

        return true_set, false_set

    def __setup_problem(self, grid, game, level):
        """Define the integer decision variables, linear constraints and linear objective function
           for solving Hexcells as an integer linear programming problem (ILP).

        Args:
            grid (grid.Grid): the level to formulate as an ILP problem.
            game (str, optional): the game in the series being solved.
            level (str, optional): set and number of the level being solved.
        """
        # Get the sets of unknown (i.e., orange) and known (i.e., blue and black) cells.
        self.__unknown = grid.unknown_cells()
        self.__known = grid.known_cells()

        # Get the constraints acting on each unknown cell.
        self.__get_constraints(grid)

        # Use the constraints to define equivalence classes.
        self.__get_classes(level)

        # Define integer decision variables for each equivalence class.
        self.__get_variables()

        # Source the GLPK solver and create a new problem to solve.
        self.__solver = pulp.GLPK_CMD(path=GLPK_PATH, msg=False, options=['--cuts'])
        self.__problem = pulp.LpProblem('HexcellsILP', pulp.LpMinimize)

        # Add the total number of remaining blue cells, grid and cell constraints to the problem.
        self.__add_remaining_constraint(grid)
        self.__add_column_constraints(grid)
        self.__add_cell_constraints(grid)

        # If the level is 6-6 from Hexcells Plus, add additional constraints.
        if level == '6-6' and game == 'Hexcells Plus':
            self.__add_plus_end_level_constraints(grid)

        # If the level is 6-6 from Hexcells Infinite, add additional constraints.
        elif level == '6-6' and game == 'Hexcells Infinite':
            self.__add_infinite_end_level_constraints(grid)

    def __get_constraints(self, grid):
        """Gets the set of constraints acting on each unknown cell.

        Args:
            grid (grid.Grid): the level to be formulated.
        """
        # Get the cell constraints from each known cell.
        self.__constraints = {}
        for cell_1 in self.__known:
            # Check that the constraint has a number.
            if cell_1.number is not None and cell_1.number != '?':
                # Add each the constraint to each of the unknown cell that it acts on.
                # 1-cell radius for black cells and 2-cell radius for blue cells.
                for cell_2 in grid.neighbours(cell_1):
                    # Make sure the cell is unknown.
                    if cell_2 is not None and cell_2.colour == Cell.ORANGE:
                        # Add the cell constraint to the unknown cell's set of constraints.
                        try:
                            self.__constraints[cell_2].add(cell_1)

                        # Create a new entry in the dictionary if the cell has not been met yet.
                        except KeyError:
                            self.__constraints[cell_2] = {cell_1}

        # Add the column constraints acting on each unknown cell.
        for constraint in grid.constraints:
            for cell in constraint.members:
                # Make sure the cell is unknown.
                if cell is not None and cell.colour == Cell.ORANGE:
                    # Add the grid constraint to the unknown cell's set of constraints.
                    try:
                        self.__constraints[cell].add(constraint)

                    # Create a new entry in the dictionary if the cell has not been met yet.
                    except KeyError:
                        self.__constraints[cell] = {constraint}

    def __get_classes(self, level):
        """Define equivalence classes of cells with the same constraints.

        Args:
            level (str): the level to be formulated as an ILP problem.
        """
        self.__rep_of = {}
        self.__class_of_rep = {}

        # If the level is one of the special cases, each cell is its own representative.
        # Using equivalence classes is not strict necessary but helps with optimisation.
        if level == '6-6':
            for cell in self.__unknown:
                self.__rep_of[cell] = cell
                self.__class_of_rep[cell] = {cell}
            return

        # Otherwise, define equivalence classes of cells that are subject to the same constraints.
        for cell in self.__unknown:
            # If consecutive or non-consecutive hint types act on a cell, it must be in its own equivalence class.
            # Even though two cells have the same constraints, hint types mean that they may not be able to switch places.
            if any(constraint.hint != 'normal' for constraint in self.__constraints[cell]):
                self.__rep_of[cell] = cell
                self.__class_of_rep[cell] = {cell}

            else:
                # Otherwise, iterate over the class representatives established so far.
                class_exists = False
                for rep in self.__class_of_rep:
                    # Check if there is already an equivalence class for the unknown cell.
                    if self.__constraints[cell] == self.__constraints[rep]:
                        # If so, add it to the existing class.
                        self.__class_of_rep[rep].add(cell)
                        self.__rep_of[cell] = rep
                        class_exists = True
                        break

                # If there is no existing class for the cell, create a new one.
                if not class_exists:
                    self.__rep_of[cell] = cell
                    self.__class_of_rep[cell] = {cell}

    def __get_variables(self):
        """Define integer decision variables for each equivalence class."""
        # Define a decision variable for each class with domain from 0 to the number of unknown cells in the class.
        self.__variables = {rep: pulp.LpVariable(f'x_{rep.grid_coords}', 0, len(rep_class), 'Integer')
                            for rep, rep_class in self.__class_of_rep.items()}

    def __get_var(self, cell):
        """A helper function to get the variable/value of a given cell.

        Args:
            cell (grid.Cell): cell to get the variable/value for.

        Returns:
            int or pulp.LpVariable: the variable/value of the cell.
        """
        # If a cell is blank or black, it is known and has a value of 0.
        if cell is None or cell.colour == Cell.BLACK:
            return 0

        # If a cell is known to be blue, it has a value of 1.
        if cell.colour == Cell.BLUE:
            return 1

        # Otherwise, the cell must be unknown.
        # If the cell is its own representative, return the variable for the equivalence class.
        if self.__rep_of[cell] is cell:
            return self.__variables[cell]

        # Otherwise, return 0 as the cell is already accounted for by its representative.
        return 0

    def __add_remaining_constraint(self, grid):
        """Add the constraint for the total number of blue cells to uncover.

        Args:
            grid (grid.Grid): the level to be formulated as an ILP problem.
        """
        # Constrain the total number of blue cells to be the equal to the number of remaining blue cells.
        self.__problem += pulp.lpSum(self.__get_var(cell) for cell in self.__unknown) == grid.remaining

    def __add_column_constraints(self, grid):
        """Add the level's grid constraints to the problem.

        Args:
            grid (grid.Grid): the level to be formulated as an ILP problem.
        """
        # Iterate over each grid constraint.
        for constraint in grid.constraints:
            # Constrain the total number of blue cells in the column to be equal to the constraint's number.
            self.__problem += pulp.lpSum(self.__get_var(cell) for cell in constraint.members) == constraint.number

            # For {n} grid constraints, cells that are at least n cells apart cannot both be blue.
            if constraint.hint == 'consecutive':
                for span in range(constraint.number, len(constraint.members)):
                    for start in range(len(constraint.members) - span):
                        self.__problem += pulp.lpSum([self.__get_var(constraint.members[start]),
                                                      self.__get_var(constraint.members[start + span])]) <= 1

            # For -n- grid constraints, any n consecutive cells in a row, column or diagonal may contain at most n-1 blue cells
            elif constraint.hint == 'non-consecutive':
                for offset in range(len(constraint.members) - constraint.number + 1):
                    self.__problem += pulp.lpSum(self.__get_var(constraint.members[offset + i])
                                                 for i in range(constraint.number)) <= constraint.number - 1

    def __add_cell_constraints(self, grid):
        """Add the level's cell constraints to the problem.

        Args:
            grid (grid.Grid): the level to be formulated as an ILP problem.
        """
        # Iterate over each known cell with a number constraint.
        for cell in self.__known:
            if cell.number is not None and cell.number != '?':
                # Get the neighbours of the cell.
                # For black cells, this is the 1-cell radius. For blue cells, it is the 2-cell radius.
                neighbours = grid.neighbours(cell)
                # The number of neighbouring blue cells must be equal to the cell number.
                self.__problem += pulp.lpSum(self.__get_var(neighbour) for neighbour in neighbours) == cell.number

                # Check if there are any hint types to consider.
                # These constraints are only valid for black cells with a number between 2 and 4,
                # or blue cells with numbers between 2 and 10.
                # Cells with hint types and numbers outside these ranges are not well-defined.
                if (cell.hint != 'normal' and
                    ((cell.colour == Cell.BLACK and 2 <= cell.number <= 4) or
                     (cell.colour == Cell.BLUE and 2 <= cell.number <= 10))):

                    # Blue cells with hint types are only used in one level in Hexcells, 4-6 from Hexcells Plus and cannot be created in custom levels.
                    # In this sole level, these constraints only apply to the outer ring of the 2-cell radius neighbourhood, rather than also including
                    # the directly adjacent neighbours). Therefore, these constraints have been implemented in the same way for the solver.
                    # In other words, functionality for blue cells with hint types is not well-defined when they have unknown adjacent cells.
                    if cell.colour == Cell.BLUE:
                        neighbours = grid.find_neighbours(cell, Grid.OUTER)

                    n = len(neighbours)
                    if cell.hint == 'consecutive':
                        # For {n} blue cells, neighbouring cells that are at least n cells apart cannot both be blue.
                        # The constraint is defined in a different way to the grid variant. Grid constraints skip gaps (i.e., empty cells can be ignored)
                        # but cell constraints do not skip gaps. Additionally, cell constraints wrap around whereas grid constraints they follow straight lines.
                        # Therefore, there may be two distances between two cells in a cell's neighbourhood: the clockwise and anti-clockwise distances.
                        # Additionally, if there are missing cells in the neighbourhood, one or more of the distances may be invalid, as the constraint does not skip gaps.
                        # The minimum distance between two cells, while accounting for gaps, calculated by __dist.
                        if cell.colour == Cell.BLUE:
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if Solver.__dist(neighbours, i, j) in range(cell.number, n):
                                        self.__problem += pulp.lpSum([self.__get_var(neighbours[i]), self.__get_var(neighbours[j])]) <= 1

                        # For {n} black cells, the following two patterns cannot occur: ``--X--'' and ``X--X'' where X denotes a blue cell and -- denotes a non-blue cell.
                        # That is, there can be no isolated blue cell or isolated gap.
                        # Note that the indexing wraps around.
                        elif cell.colour == Cell.BLACK:
                            for i in range(n):
                                cond = (self.__get_var(neighbours[i]) -
                                        self.__get_var(neighbours[(i + 1) % n]) -
                                        self.__get_var(neighbours[(i - 1) % n]))

                                self.__problem += -1 <= cond <= 0

                    # Similarly to grid constraints, for -n- cells (either blue or black), a consecutive block of n neighbouring cells can contain at most n-1 blue cells.
                    if cell.hint == 'non-consecutive':
                        for i in range(n):
                            if all(neighbours[(i + j) % n] is not None for j in range(cell.number - 1)):
                                self.__problem += pulp.lpSum(self.__get_var(neighbours[(i + j) % n])
                                                             for j in range(cell.number)) <= cell.number - 1

    @staticmethod
    def __dist(neighbours, i, j):
        """Calculate the circular distance between two cells while accounting for gaps.

        Args:
            neighbours (list): the set of neighbour cells to calculate the distance over.
            i (int): index of the first cell.
            j (int): index of the second cell.

        Returns:
            int: the minimum circular distance between the two cells.
        """
        n = len(neighbours)

        # Calculate the clockwise distance between the cells.
        dist_1 = 0
        for k in range(i + 1, j + 1):
            # Check that gaps are accounted for.
            if neighbours[k % n] is not None:
                dist_1 += 1
            else:
                # If there is a gap, the distance is infinite in this direction.
                dist_1 = float('inf')
                break

        # Calculate the anti-clockwise distance between the cells.
        dist_2 = 0
        for k in range(i - 1, j - n - 1, -1):
            # Check that gaps are accounted for.
            if neighbours[k % n] is not None:
                dist_2 += 1
            else:
                # If there is a gap, the distance is infinite in this direction.
                dist_2 = float('inf')
                break

        # Return the minimum of the two distances.
        return min(dist_1, dist_2)

    def __add_plus_end_level_constraints(self, grid):
        """Add the constraints from the final level of Hexcells Plus to the problem formulation.

        Args:
            grid (grid.Grid): the level to be formulated as an ILP problem.
        """
        # Get the cells in each letter of "FINISH" using the columns of each letter.
        letters = [[0, 1, 2], # F
                   [5], # I
                   [8, 9, 10, 11, 12], # N
                   [15], # I
                   [18, 19, 20], # S
                   [23, 24, 25, 26]] # H

        # Combine the cells from columns of the same letter.
        for i, letter in enumerate(letters):
            cells = []
            for col in letter:
                cells.extend([self.__get_var(cell) for cell in grid.get_column(col)])

            letters[i] = cells

        # Matching letters (i.e., the Is) contain 5 blue cells in total.
        # Also, the first and last letters (F and H) have 16 blue cells between them.
        self.__problem += pulp.lpSum(letters[1] + letters[3]) == 5
        self.__problem += pulp.lpSum(letters[0] + letters[5]) == 16

        # One of the letters contains no blue cells.
        xs = []
        for i, letter in enumerate(letters):
            # Create a binary variable for each letter.
            x = pulp.LpVariable(f'x_{i}', 0, 1, 'Integer')
            # If x is 0 for the letter, it cannot contain any blue cells.
            self.__problem += pulp.lpSum(letter) <= x * len(letter)
            xs.append(x)

        # Constrain at least one of the letters to have x=0 (i.e., no blue cells).
        self.__problem += pulp.lpSum(xs) == len(letters) - 1

    def __add_infinite_end_level_constraints(self, grid):
        """Add the constraints from the final level of Hexcells Infinite to the problem formulation.

        Args:
            grid (grid.Grid): the level to be formulated as an ILP problem.
        """
        # The first constraint is that all columns contain an odd number of blue cells.
        for col in range(grid.cols):
            # Get the cells in the column.
            column = grid.get_column(col)

            # Define a new decision variable with domain from 0 to half of the number of cells in the column.
            # Use the variable to constrain the number of cells to be odd.
            x = pulp.LpVariable(f'x_{col}', 0, (len(column) - 1) // 2, 'Integer')
            self.__problem += pulp.lpSum(self.__get_var(cell) for cell in column) == 2 * x + 1

        # The second constraint applies to the 2-cell neighbours of two cells.
        centre_1, centre_2 = grid[8, 4], grid[8, 12]
        region_1 = grid.find_neighbours(centre_1, Grid.COMBINED) + [centre_1]
        region_2 = grid.find_neighbours(centre_2, Grid.COMBINED) + [centre_2]

        # Constrain each set of cells to contain exactly 7 blue cells.
        self.__problem += pulp.lpSum(self.__get_var(cell) for cell in region_1) == 7
        self.__problem += pulp.lpSum(self.__get_var(cell) for cell in region_2) == 7
