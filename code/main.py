import tkinter as tk 
from navigate import Navigator

class GUI:
    def __init__(self):
        self.__menu = Navigator()
        self.__root = tk.Tk()
        self.__root.title('HexSolver')
        self.__root.resizable(width=False, height=False)
        self.__window_width = 300
        self.__window_height = 600 
        self.__root.geometry('{0}x{1}+0+0'.format(self.__window_width, self.__window_height))
        self.__create_solver_frame()
     
    def start(self):
        tk.mainloop()
     
    def __create_solver_frame(self):
        self.__solve_var = tk.IntVar(self.__root)
        self.__solve_var.set(0)
        self.__solve_var.trace('w', self.__solver_radiobutton_callback)
        
        self.__solve_label = tk.Label(self.__root,
                                      font=('Arial Black', 10),
                                      text='-'*80+'\n'+'Solver Options\n'+'-'*80)
        
        self.__level_radiobutton = tk.Radiobutton(self.__root,
                                                  variable=self.__solve_var,
                                                  value=0,
                                                  text='Specific Level',
                                                  font=('Arial Black', 9))
        
        self.__set_radiobutton = tk.Radiobutton(self.__root,
                                                variable=self.__solve_var,
                                                value=1,
                                                text='Specific Set',                                                
                                                font=('Arial Black', 9))
                
        self.__game_radiobutton = tk.Radiobutton(self.__root,
                                                 variable=self.__solve_var,
                                                 value=2,
                                                 text='Entire Game',                                                 
                                                 font=('Arial Black', 9))
        
        self.__generator_radiobutton = tk.Radiobutton(self.__root,
                                                      variable=self.__solve_var, 
                                                      value=3,
                                                      text='Level Generator',                                                      
                                                      font=('Arial Black', 9))
                
        self.__solve_label.grid(sticky='w', row=0, column=0, columnspan=2)
        self.__level_radiobutton.grid(sticky='w', row=1, column=0)
        self.__set_radiobutton.grid(sticky='w', row=2, column=0)
        self.__game_radiobutton.grid(sticky='w', row=3, column=0)
        self.__generator_radiobutton.grid(sticky='w', row=4, column=0)
        
        sets = [str(x) for x in range(1, 7)]
        self.__set_var = tk.StringVar(self.__root)
        self.__set_var.set('-')
        self.__set_var.trace('w', self.__check_ready_to_solve)
        
        self.__set_frame = tk.Frame(self.__root)
        self.__set_label = tk.Label(self.__set_frame,
                                    font=('Arial', 9),
                                    text='Select Set:')
        
        self.__set_optionmenu = tk.OptionMenu(self.__set_frame,
                                              self.__set_var,
                                              *sets)
        self.__set_label.pack(side='left')
        self.__set_optionmenu.pack()
        
        levels = [str(x) for x in range(1, 7)]
        self.__level_var = tk.StringVar(self.__root)
        self.__level_var.set('-')
        self.__level_var.trace('w', self.__check_ready_to_solve)
        
        self.__level_frame = tk.Frame(self.__root)
        self.__level_label = tk.Label(self.__level_frame,
                                      font=('Arial', 9),
                                      text='Select Level:')
        
        self.__level_optionmenu = tk.OptionMenu(self.__level_frame,
                                                self.__level_var,
                                                *levels)
        self.__level_label.pack(side='left')
        self.__level_optionmenu.pack()
        self.__set_frame.grid(sticky='w', row=1, column=1)
        self.__level_frame.grid(sticky='w', row=2, column=1)
        
        self.__solve_button = tk.Button(self.__root,
                                        text='Solve',
                                        state=tk.DISABLED,
                                        font=('Arial', 10),
                                        command=self.__solve)
        
        self.__solve_button.grid(sticky='nesw', row=5, column=0, columnspan=2, pady=(10, 0))
        
    def __solver_radiobutton_callback(self, *args):
        if self.__solve_var.get() == 0:
            self.__set_optionmenu.configure(state=tk.NORMAL)
            self.__level_optionmenu.configure(state=tk.NORMAL)
            
        elif self.__solve_var.get() == 1:
            self.__level_var.set('-')
            self.__set_optionmenu.configure(state=tk.NORMAL)
            self.__level_optionmenu.configure(state=tk.DISABLED)
            
        else:
            self.__set_var.set('-')
            self.__level_var.set('-')
            self.__set_optionmenu.configure(state=tk.DISABLED)
            self.__level_optionmenu.configure(state=tk.DISABLED)
            
        self.__check_ready_to_solve()
            
    def __check_ready_to_solve(self, *args):
        if ((self.__solve_var.get() == 0 and (self.__set_var.get() == '-' or self.__level_var.get() == '-')) or
            (self.__solve_var.get() == 1 and self.__set_var.get() == '-')):
            self.__solve_button.configure(state=tk.DISABLED)
        else:
            self.__solve_button.configure(state=tk.NORMAL)
      
    def __solve(self):
        if self.__solve_var.get() == 0:
            self.__menu.solve_level(self.__set_var.get()+'-'+self.__level_var.get())
        elif self.__solve_var.get() == 1:
            self.__menu.solve_set(self.__set_var.get())
        elif self.__solve_var.get() == 2:
            self.__menu.solve_game()
        elif self.__solve_var.get() == 3:
            self.__menu.puzzle_generator()
         
if __name__ == '__main__':
    application = GUI()
    application.start()
    
    #menu = Navigator()
    #menu.solve(continuous=True)
    
    #menu.puzzle_generator()
    
    #menu.save_slot(2)
    #menu.solve_level('6-6')
    #menu.solve_world('6')
    #menu.solve_game()
