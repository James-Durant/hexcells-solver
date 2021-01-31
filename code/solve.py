from pulp import GLPK_CMD, LpProblem, LpMinimize, LpVariable, lpSum, value
from parse import Cell
import time

class Solver:
    @staticmethod
    def __get_constraints(grid, known):
        #Map unknown cells to all relevant constraints of known cells that they are a member of.
        constraints = {}
        for cell1 in known:
            if cell1.digit != '?':
                for cell2 in grid.neighbours(cell1):
                    if cell2 != None and cell2.colour == Cell.ORANGE:
                        try:
                            constraints[cell2].add(cell1)
                        except:
                            constraints[cell2] = set([cell1])
 
        for constraint in grid.constraints:
            for cell in constraint.members:
                try:
                    constraints[cell].add(constraint)
                except:
                    constraints[cell] = set([constraint])
    
        return constraints
    
    @staticmethod
    def __get_reps(unknown, constraints):
        rep_of = {}
        for cell1 in unknown:
            if cell1 in constraints:
                for cell2 in unknown:
                    if cell2 in constraints and constraints[cell1] == constraints[cell2]:
                        rep_of[cell1] = cell2
            else:
                rep_of[cell1] = cell1
        
        for cell in constraints:
            for constraint in constraints[cell]:
                if constraint.hint != 'normal':
                    rep_of[cell] = cell
                
        return rep_of
    
    @staticmethod
    def __get_classes(unknown, rep_of):
        classes = {}
        for rep in unknown:
            if rep_of[rep] is rep:
                classes[rep] = sum(1 for cell in unknown if rep_of[cell] is rep)
        return classes
    
    @staticmethod
    def __get_var(cell, rep_of, variables):
        if cell is None or cell.colour == Cell.BLACK:
            return 0
        if cell.colour == Cell.BLUE:
            return 1
        if rep_of[cell] is cell:
            return variables[cell]
        else: # cell is non representative
            return 0
        
    @staticmethod  
    def __get_true_false_classes(classes, rep_of, variables):
        true_set  = set()
        false_set = set()
        
        for rep, size in classes.items():
            var = Solver.__get_var(rep, rep_of, variables)
            
            if value(var) == 0:
                false_set.add(rep)
            elif value(var) == size:
                true_set.add(rep)
                
        return true_set, false_set
    
    @staticmethod 
    def solve(parser):
        grid = parser.parse_grid()
        
        while True:
            clicked_cells = Solver.__solve_single_step(parser.window, grid)
            if len(clicked_cells)-len(grid.unknown_cells()) == 0:
                break
        
            time.sleep(1)
            grid.remaining = parser.parse_clicked_cells(clicked_cells)
        
    @staticmethod
    def __solve_single_step(window, grid):
        #print(grid)
        unknown     = grid.unknown_cells()
        known       = grid.known_cells()
        constraints = Solver.__get_constraints(grid, known)
        rep_of      = Solver.__get_reps(unknown, constraints)
        classes     = Solver.__get_classes(unknown, rep_of)

        solver    = GLPK_CMD(path=r'C:\Users\james\Documents\winglpk-4.65\glpk-4.65\w64\glpsol.exe', msg=False, options=['--cuts'])
        problem   = LpProblem('HexcellsMILP', LpMinimize)
        variables = {rep: LpVariable(str(rep.grid_coords), 0, size, 'Integer') for rep, size in classes.items()}

        # The number of remaining blue cells is known
        problem += lpSum(Solver.__get_var(cell, rep_of, variables) for cell in unknown) == grid.remaining
        
        # Constraints from column number information
        for constraint in grid.constraints:
            # The sum of all cells in that column is the column value
            problem += lpSum(Solver.__get_var(cell, rep_of, variables) for cell in constraint.members) == constraint.size
            
            # Additional information (together/seperated) available?
            if constraint.hint == 'consecutive':
                # For {n}: cells that are at least n appart cannot be both blue.
                # Example: For {3}, the configurations X??X, X???X, X????X, ... are impossible.
                for span in range(constraint.size, len(constraint.members)):
                    for start in range(len(constraint.members)-span):
                        problem += lpSum([Solver.__get_var(constraint.members[start], rep_of, variables), Solver.__get_var(constraint.members[start+span], rep_of, variables)]) <= 1
                        
            elif constraint.hint == 'non-consecutive':
                # For -n-, the sum of any range of n cells may contain at most n-1 blues
                for offset in range(len(constraint.members)-constraint.size+1):
                    problem += lpSum(Solver.__get_var(constraint.members[offset+i], rep_of, variables) for i in range(constraint.size)) <= constraint.size-1
    
        # Constraints from cell number information
        for cell in known:
            if cell.digit != None and cell.digit != '?':       
                neighbours = grid.neighbours(cell)
                problem += lpSum(Solver.__get_var(neighbour, rep_of, variables) for neighbour in neighbours) == cell.digit
                
                # Additional togetherness information available. Only relevant if value between 2 and 4.
                if cell.hint != 'normal' and 2 <= cell.digit <= 4:
                    m = neighbours + neighbours #Have array wrap around.
                    
                    if cell.hint == 'consecutive':
                        # two patterns not occuring: "-X-" and the "X-X": No lonely blue cell and no lonely gap
                        for i in range(len(neighbours)):
                            # No lonely cell condition:
                            # Say m[i] is a blue.
                            # Then m[i-1] or m[i+1] must be blue.
                            # That means: -m[i-1] +m[i] -m[i+1] <= 0
                            # Note that m[i+1] and m[i-1] only count
                            # if they are real neighbours.
                            cond = Solver.__get_var(m[i], rep_of, variables)
                            cond -= Solver.__get_var(m[i-1], rep_of, variables)
                            cond -= Solver.__get_var(m[i+1], rep_of, variables)
                                
                            # no isolated cell
                            problem += cond <= 0
                            # no isolated gap (works by a similar argument)
                            problem += cond >= -1
                            
                    elif cell.hint == 'non-consecutive':
                        # any circular range of n cells contains at most n-1 blues.
                        for i in range(len(neighbours)):
                            # the range m[i], ..., m[i+n-1] may not all be blue if they are consecutive
                            if all(m[i+j+1] != None for j in range(cell.digit-1)):
                                problem += lpSum(Solver.__get_var(m[i+j], rep_of, variables) for j in range(cell.digit)) <= cell.digit-1
                
        # First, get any solution.
        # Default solver can't handle no objective, so invent one:
        spam = LpVariable('spam', 0, 1, 'binary')
        problem += (spam == 1)
        problem.setObjective(spam) # no optimisation function yet
        problem.solve(solver)
                
        true_class, false_class = Solver.__get_true_false_classes(classes, rep_of, variables)
        
        clicked_cells = []
        while true_class or false_class:
            # Now try to vary as much away from the
            # initial solution as possible:
            # We try to make the variables True, that were False before
            # and vice versa. If no change could be achieved, then
            # the remaining variables have their unique possible value.
            problem.setObjective(lpSum(Solver.__get_var(t, rep_of, variables) for t in true_class) - lpSum(Solver.__get_var(f, rep_of, variables) for f in false_class))
            problem.solve(solver)
            
            # all true variables stayed true and false stayed false?
            # Then they have their unique value and we are done!
            if value(problem.objective) == sum(classes[rep] for rep in true_class):
                for tf_set, kind in [(true_class, "left"), (false_class, "right")]:
                    for rep in tf_set:
                        for cell in unknown:
                            if rep_of[cell] is rep:
                                window.click_cell(cell, kind)
                                clicked_cells.append(cell)
                                
                return clicked_cells
            
            true_new, false_new = Solver.__get_true_false_classes(classes, rep_of, variables)
            
            # remember only those classes that subbornly kept their pure trueness/falseness
            true_class  &= true_new
            false_class &= false_new
            
        raise RuntimeError('solver failed to finish puzzle')