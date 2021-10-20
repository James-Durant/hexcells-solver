from pulp import GLPK_CMD, LpProblem, LpMinimize, LpVariable, lpSum, value
from parse import Cell

class Solver:
    __GLPK_PATH = 'resources/winglpk-4.65/glpk-4.65/w64/glpsol.exe'

    def __init__(self, parser):
        self.__parser = parser

    def solve(self, level=None, game=None):
        grid = self.__parser.parse_grid()

        while True:
            left_click_cells, right_click_cells = self.solve_single_step(grid, game, level)
            if len(left_click_cells)+len(right_click_cells)-len(self.__unknown) == 0:
                self.__parser.click_cells(left_click_cells, 'left')
                self.__parser.click_cells(right_click_cells, 'right')
                break

            _, remaining = self.__parser.parse_clicked(grid, left_click_cells, right_click_cells)
            if remaining is not None:
                grid.remaining = remaining

    def solve_single_step(self, grid, game=None, level=None):
        self.__setup_problem(grid, game, level)

        left_click_cells, right_click_cells = [], []
        true_class, false_class = self.__get_true_false_classes()
        while true_class or false_class:
            true_sum  = lpSum(self.__get_var(true) for true in true_class)
            false_sum = lpSum(self.__get_var(false) for false in false_class)

            self.__problem.setObjective(true_sum-false_sum)
            self.__problem.solve(self.__solver)

            if value(self.__problem.objective) == sum(self.__classes[rep] for rep in true_class):
                for tf_set, kind in [(true_class, 'left'), (false_class, 'right')]:
                    for rep in tf_set:
                        for cell in self.__unknown:
                            if self.__rep_of[cell] is rep:
                                if kind == 'left':
                                    left_click_cells.append(cell)
                                elif kind == 'right':
                                    right_click_cells.append(cell)

                return left_click_cells, right_click_cells

            true_new, false_new = self.__get_true_false_classes()

            true_class &= true_new
            false_class &= false_new

        raise RuntimeError('solver failed to finish puzzle')

    def __setup_problem(self, grid, game, level):
        self.__unknown = grid.unknown_cells()
        self.__known = grid.known_cells()

        self.__get_constraints(grid)
        self.__get_reps(level)
        self.__get_classes()
        self.__get_variables()

        self.__solver = GLPK_CMD(path=Solver.__GLPK_PATH, msg=False, options=['--cuts'])
        self.__problem = LpProblem('HexcellsMILP', LpMinimize)

        self.__add_remaining_constraint(grid)
        self.__add_column_constraints(grid)
        self.__add_cell_constraints(grid)

        if level == '6-6' and game == 'Hexcells Plus':
            self.__add_plus_end_level_constraints(grid)

        elif level == '6-6' and game == 'Hexcells Infinite':
            self.__add_infinite_end_level_constraints(grid)

        temp = LpVariable('temp', 0, 1, 'binary')
        self.__problem += temp == 1
        self.__problem.setObjective(temp)
        self.__problem.solve(self.__solver)

    def __add_infinite_end_level_constraints(self, grid):
        for col in range(grid.cols):
            column = grid.get_column(col)
            x = LpVariable('x_'+str(col), 0, (len(column)-1)//2, 'Integer')
            self.__problem += lpSum(self.__get_var(cell) for cell in column) == 2*x+1

        centre_1, centre_2 = grid[8, 4], grid[8, 12]

        inner_1 = grid.flower_neighbours(centre_1) + [centre_1]
        inner_2 = grid.flower_neighbours(centre_2) + [centre_2]

        self.__problem += lpSum(self.__get_var(cell) for cell in inner_1) == 7
        self.__problem += lpSum(self.__get_var(cell) for cell in inner_2) == 7

    def __add_plus_end_level_constraints(self, grid):
        letters = [[0, 1, 2],
                   [5],
                   [8, 9, 10, 11, 12],
                   [15],
                   [18, 19, 20],
                   [23, 24, 25, 26]]

        for i, letter in enumerate(letters):
            cells = []
            for col in letter:
                cells.extend([self.__get_var(cell) for cell in grid.get_column(col)])

            letters[i] = cells

        self.__problem += lpSum(letters[1]+letters[3]) == 5
        self.__problem += lpSum(letters[0]+letters[5]) == 16

        xs = []
        for i, letter in enumerate(letters):
            x = LpVariable('x_'+str(i), 0, 1, 'Integer')
            self.__problem += lpSum(letter) <= x*len(letter)
            xs.append(x)

        self.__problem += lpSum(xs) == len(letters)-1

    def __get_constraints(self, grid):
        self.__constraints = {}
        for cell1 in self.__known:
            if cell1.digit != '?':
                for cell2 in grid.neighbours(cell1):
                    if cell2 != None and cell2.colour == Cell.ORANGE:
                        try:
                            self.__constraints[cell2].add(cell1)
                        except:
                            self.__constraints[cell2] = set([cell1])

        for constraint in grid.constraints:
            for cell in constraint.members:
                try:
                    self.__constraints[cell].add(constraint)
                except:
                    self.__constraints[cell] = set([constraint])

    def __get_reps(self, level):
        self.__rep_of = {}
        if level == '6-6':
            for cell in self.__unknown:
                self.__rep_of[cell] = cell
            return

        self.__rep_of = {}
        for cell1 in self.__unknown:
            if cell1 in self.__constraints:
                for cell2 in self.__unknown:
                    if cell2 in self.__constraints and self.__constraints[cell1] == self.__constraints[cell2]:
                        self.__rep_of[cell1] = cell2
            else:
                self.__rep_of[cell1] = cell1

        for cell in self.__constraints:
            if any(constraint.hint != 'normal' for constraint in self.__constraints[cell]):
                self.__rep_of[cell] = cell

    def __get_classes(self):
        self.__classes = {}
        for rep in self.__unknown:
            if self.__rep_of[rep] is rep:
                self.__classes[rep] = sum(1 for cell in self.__unknown if self.__rep_of[cell] is rep)

    def __get_variables(self):
        self.__variables = {}
        for rep, size in self.__classes.items():
            self.__variables[rep] = LpVariable(str(rep.grid_coords), 0, size, 'Integer')

    def __add_remaining_constraint(self, grid):
        self.__problem += lpSum(self.__get_var(cell) for cell in self.__unknown) == grid.remaining

    def __add_column_constraints(self, grid):
        for constraint in grid.constraints:
            self.__problem += lpSum(self.__get_var(cell) for cell in constraint.members) == constraint.size

            if constraint.hint == 'consecutive':
                for span in range(constraint.size, len(constraint.members)):
                    for start in range(len(constraint.members)-span):
                        self.__problem += lpSum([self.__get_var(constraint.members[start]), self.__get_var(constraint.members[start+span])]) <= 1

            elif constraint.hint == 'non-consecutive':
                for offset in range(len(constraint.members)-constraint.size+1):
                    self.__problem += lpSum(self.__get_var(constraint.members[offset+i]) for i in range(constraint.size)) <= constraint.size-1

    def __add_cell_constraints(self, grid):
        for cell in self.__known:
            if cell.digit != None and cell.digit != '?':
                neighbours = grid.neighbours(cell)
                self.__problem += lpSum(self.__get_var(neighbour) for neighbour in neighbours) == cell.digit

                if (cell.hint != 'normal' and
                   ((cell.colour == Cell.BLACK and 2 <= cell.digit <= 4) or
                   (cell.colour == Cell.BLUE and 2 <= cell.digit <= 10))):

                    if cell.colour == Cell.BLUE:
                        neighbours = grid.outer_neighbours(cell)

                    n = len(neighbours)
                    if cell.hint == 'consecutive':
                        if cell.colour == Cell.BLUE:
                            for i in range(n):
                                for j in range(i+1, n):
                                    if self.__dist(neighbours, i, j) in range(cell.digit, n):
                                        self.__problem += lpSum([self.__get_var(neighbours[i]), self.__get_var(neighbours[j])]) <= 1

                        elif cell.colour == Cell.BLACK:
                            for i in range(n):
                                cond = self.__get_var(neighbours[i]) - self.__get_var(neighbours[(i+1)%n]) - self.__get_var(neighbours[(i-1)%n])
                                self.__problem += -1 <= cond <= 0

                    if cell.hint == 'non-consecutive':
                        for i in range(n):
                            if all(neighbours[(i+j+1)%n] != None for j in range(cell.digit-1)):
                                self.__problem += lpSum(self.__get_var(neighbours[(i+j)%n]) for j in range(cell.digit)) <= cell.digit-1

    def __dist(self, neighbours, i, j):
        dist1 = 0
        for k in range(i+1, j+1):
            if neighbours[k] != None:
                dist1 += 1
            else:
                dist1 = float('inf')
                break

        dist2 = 0
        n = len(neighbours)
        for k in range(i-1, j-n-1, -1):
            if neighbours[k%n] != None:
                dist2 += 1
            else:
                dist2 = float('inf')
                break

        return min(dist1, dist2)

    def __get_var(self, cell):
        if cell is None or cell.colour == Cell.BLACK:
            return 0
        if cell.colour == Cell.BLUE:
            return 1
        if self.__rep_of[cell] is cell:
            return self.__variables[cell]
        return 0

    def __get_true_false_classes(self):
        true_set, false_set = set(), set()

        for rep, size in self.__classes.items():
            val = value(self.__get_var(rep))

            if val == 0:
                false_set.add(rep)
            elif val == size:
                true_set.add(rep)

        return true_set, false_set
