from pulp import GLPK_CMD, LpProblem, LpMinimize, LpVariable, lpSum, value
from parse import Cell
import time

class Solver:
    __GLPK_PATH = r'C:\Users\james\Documents\winglpk-4.65\glpk-4.65\w64\glpsol.exe'
    
    def __init__(self, parser):
        self.__parser = parser

    def solve(self):
        self.__grid = self.__parser.parse_grid()
        
        while True:
            self.__setup_problem()
            clicked_cells = self.__solve_single_step()
            if len(clicked_cells)-len(self.__grid.unknown_cells()) == 0:
                break
        
            time.sleep(1)
            self.__grid.remaining = self.__parser.parse_clicked_cells(clicked_cells)
    
    def __setup_problem(self):
        self.__unknown = self.__grid.unknown_cells()
        self.__known = self.__grid.known_cells()
        
        self.__get_constraints()
        self.__get_reps()
        self.__get_classes()
        self.__get_variables()
        
        self.__solver = GLPK_CMD(path=Solver.__GLPK_PATH, msg=False, options=['--cuts'])
        self.__problem = LpProblem('HexcellsMILP', LpMinimize)
        
        self.__add_remaining_constraint()
        self.__add_column_constraints()
        self.__add_cell_constraints()
        
        temp = LpVariable('spam', 0, 1, 'binary')
        self.__problem += (temp == 1)
        self.__problem.setObjective(temp)
        self.__problem.solve(self.__solver)
    
    def __solve_single_step(self):   
        clicked_cells = []
        true_class, false_class = self.__get_true_false_classes()
        while true_class or false_class:
            true_sum  = lpSum(self.__get_var(true)  for true  in true_class)
            false_sum = lpSum(self.__get_var(false) for false in false_class)
            
            self.__problem.setObjective(true_sum-false_sum)
            self.__problem.solve(self.__solver)
            
            if value(self.__problem.objective) == sum(self.__classes[rep] for rep in true_class):
                for tf_set, kind in [(true_class, "left"), (false_class, "right")]:
                    for rep in tf_set:
                        for cell in self.__unknown:
                            if self.__rep_of[cell] is rep:
                                self.__parser.window.click_cell(cell, kind)
                                clicked_cells.append(cell)
                                
                return clicked_cells
            
            true_new, false_new = self.__get_true_false_classes()

            true_class  &= true_new
            false_class &= false_new
            
        raise RuntimeError('solver failed to finish puzzle')
    
    def __get_constraints(self):
        self.__constraints = {}
        for cell1 in self.__known:
            if cell1.digit != '?':
                for cell2 in self.__grid.neighbours(cell1):
                    if cell2 != None and cell2.colour == Cell.ORANGE:
                        try:
                            self.__constraints[cell2].add(cell1)
                        except:
                            self.__constraints[cell2] = set([cell1])
 
        for constraint in self.__grid.constraints:
            for cell in constraint.members:
                try:
                    self.__constraints[cell].add(constraint)
                except:
                    self.__constraints[cell] = set([constraint])
    
    def __get_reps(self):
        self.__rep_of = {}
        for cell1 in self.__unknown:
            if cell1 in self.__constraints:
                for cell2 in self.__unknown:
                    if cell2 in self.__constraints and self.__constraints[cell1] == self.__constraints[cell2]:
                        self.__rep_of[cell1] = cell2
            else:
                self.__rep_of[cell1] = cell1
        
        for cell in self.__constraints:
            for constraint in self.__constraints[cell]:
                if constraint.hint != 'normal':
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
    
    def __add_remaining_constraint(self):
        self.__problem += lpSum(self.__get_var(cell) for cell in self.__unknown) == self.__grid.remaining
    
    def __add_column_constraints(self):
        for constraint in self.__grid.constraints:
            self.__problem += lpSum(self.get_var(cell) for cell in constraint.members) == constraint.size
            
            if constraint.hint == 'consecutive':
                for span in range(constraint.size, len(constraint.members)):
                    for start in range(len(constraint.members)-span):
                        self.__problem += lpSum([self.__get_var(constraint.members[start]), self.__get_var(constraint.members[start+span])]) <= 1
                        
            elif constraint.hint == 'non-consecutive':
                for offset in range(len(constraint.members)-constraint.size+1):
                    self.__problem += lpSum(self.__get_var(constraint.members[offset+i]) for i in range(constraint.size)) <= constraint.size-1
    
    def __add_cell_constraints(self):
        for cell in self.__known:
            if cell.digit != None and cell.digit != '?':       
                neighbours = self.__grid.neighbours(cell)
                self.__problem += lpSum(self.__get_var(neighbour) for neighbour in neighbours) == cell.digit
                if cell.hint != 'normal' and 2 <= cell.digit <= 4:
                    m = neighbours + neighbours
                    
                    if cell.hint == 'consecutive':
                        for i in range(len(neighbours)):
                            cond  = self.__get_var(m[i])
                            cond -= self.__get_var(m[i-1])
                            cond -= self.__get_var(m[i+1])
                                
                            self.__problem += cond <= 0
                            self.__problem += cond >= -1
                            
                    elif cell.hint == 'non-consecutive':
                        for i in range(len(neighbours)):
                            if all(m[i+j+1] != None for j in range(cell.digit-1)):
                                self.__problem += lpSum(self.__get_var(m[i+j]) for j in range(cell.digit)) <= cell.digit-1
   
    def __get_var(self, cell):
        if cell is None or cell.colour == Cell.BLACK:
            return 0
        if cell.colour == Cell.BLUE:
            return 1
        if self.__rep_of[cell] is cell:
            return self.__variables[cell]
        else:
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
