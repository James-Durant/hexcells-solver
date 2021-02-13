from window import Window, get_window
from parse import Parser
from solve import Solver

if __name__ == "__main__":
    window = get_window()
    parser = Parser(window)
    solver = Solver(parser)
    solver.solve()
