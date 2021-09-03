import tkinter as tk 
from subprocess import Popen
from navigate import Navigator

class GUI:
    GAMEIDS = {'Hexcells': '265890',
               'Hexcells Plus': '271900',
               'Hexcells Infinite': '304410'}
    
    def __init__(self):
        self.__root = tk.Tk()
        self.__root.title('HexSolver')
        self.__root.resizable(width=False, height=False)
        self.__window_width = 300
        self.__window_height = 600 
        self.__root.geometry('{0}x{1}+0+0'.format(self.__window_width, self.__window_height))
        
        self.__create_info_frame()
        self.__create_solver_frame()
        
        self.__check_game_running()
        
        tk.mainloop()
        
    def __check_game_running(self):
        try:
            self.__menu = Navigator()
            self.__update_status('game running')
            self.__game_var.set(self.__menu.window.title)
            
        except RuntimeError: # Make custom error
            self.__menu = None
            self.__update_status('game not running')
            self.__game_var.set(-1)
       
    def __update_status(self, status):
        self.__status_label.configure(text='Status: {}'.format(status)) 
       
    def __load_game(self):
        if self.__menu:
            self.__menu.close_game()
            
        process = Popen([r'C:\Program Files (x86)\Steam\steam.exe', r'steam://rungameid/'+GUI.GAMEIDS[self.__game_var.get()]],
                        shell=True, stdin=None, stdout=None, stderr=None, close_fds=True) 
        process.wait()
        
        while True:
            try:
                self.__menu = Navigator()
                self.__update_status('game running')
                break
            except RuntimeError:
                self.__update_status('game not running')
       
    def __create_info_frame(self):
        self.__info_frame = tk.Frame(self.__root)
        
        self.__info_label = tk.Label(self.__info_frame,
                                     font=('Arial Black', 10),
                                     text='-'*80+'\n'+'Game Options\n'+'-'*80)
        
        self.__game_var = tk.StringVar(self.__root)
        self.__game_var.set(-1)
        
        self.__game_1_radiobutton = tk.Radiobutton(self.__info_frame,
                                                   variable=self.__game_var,
                                                   value='Hexcells',
                                                   text='Hexcells',
                                                   font=('Arial Black', 9),
                                                   command=self.__load_game)
        
        self.__game_2_radiobutton = tk.Radiobutton(self.__info_frame,
                                                   variable=self.__game_var,
                                                   value='Hexcells Plus',
                                                   text='Hexcells Plus',                                                
                                                   font=('Arial Black', 9),
                                                   command=self.__load_game)
                
        self.__game_3_radiobutton = tk.Radiobutton(self.__info_frame,
                                                   variable=self.__game_var,
                                                   value='Hexcells Infinite',
                                                   text='Hexcells Infinite',                                                 
                                                   font=('Arial Black', 9),
                                                   command=self.__load_game)
        
        self.__status_label = tk.Label(self.__info_frame,
                             font=('Arial', 9))
        
        self.__info_label.grid(sticky='w', row=0, column=0)
        self.__game_1_radiobutton.grid(sticky='w', row=1, column=0)
        self.__game_2_radiobutton.grid(sticky='w', row=2, column=0)
        self.__game_3_radiobutton.grid(sticky='w', row=3, column=0)
        self.__status_label.grid(sticky='w', row=4, column=0)
        
        self.__info_frame.grid(row=0, column=0)
        
    def __create_solver_frame(self):
        self.__solver_frame = tk.Frame(self.__root)
        
        self.__solve_var = tk.IntVar(self.__root)
        self.__solve_var.set(0)
        self.__solve_var.trace('w', self.__solver_radiobutton_callback)
        
        self.__solve_label = tk.Label(self.__solver_frame,
                                      font=('Arial Black', 10),
                                      text='-'*80+'\n'+'Solver Options\n'+'-'*80)
        
        self.__level_radiobutton = tk.Radiobutton(self.__solver_frame,
                                                  variable=self.__solve_var,
                                                  value=0,
                                                  text='Specific Level',
                                                  font=('Arial Black', 9))
        
        self.__set_radiobutton = tk.Radiobutton(self.__solver_frame,
                                                variable=self.__solve_var,
                                                value=1,
                                                text='Specific Set',                                                
                                                font=('Arial Black', 9))
                
        self.__game_radiobutton = tk.Radiobutton(self.__solver_frame,
                                                 variable=self.__solve_var,
                                                 value=2,
                                                 text='Entire Game',                                                 
                                                 font=('Arial Black', 9))
        
        self.__generator_radiobutton = tk.Radiobutton(self.__solver_frame,
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
        
        self.__set_frame = tk.Frame(self.__solver_frame)
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
        
        self.__level_frame = tk.Frame(self.__solver_frame)
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
        
        self.__solve_button = tk.Button(self.__solver_frame,
                                        text='Solve',
                                        state=tk.DISABLED,
                                        font=('Arial', 10),
                                        command=self.__solve)
        
        self.__solve_button.grid(sticky='nesw', row=5, column=0, columnspan=2, pady=(10, 0))
        
        self.__solver_frame.grid(row=1, column=0)
        
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
    
    #menu = Navigator()
    #menu.solve(continuous=True)
    
    #menu.puzzle_generator()
    
    #menu.save_slot(2)
    #menu.solve_level('6-6')
    #menu.solve_world('6')
    #menu.solve_game()
