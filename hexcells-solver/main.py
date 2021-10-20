import tkinter as tk
from subprocess import Popen
from navigate import Navigator

GAMEIDS = {'Hexcells': '265890',
           'Hexcells Plus': '271900',
           'Hexcells Infinite': '304410'}

class GUI:
    def __init__(self):
        self.__root = tk.Tk()
        self.__root.title('HexSolver')
        self.__root.resizable(width=False, height=False)
        #self.__window_width = 300
        #self.__window_height = 600
        #self.__root.geometry('{0}x{1}+0+0'.format(self.__window_width, self.__window_height))

        self.__create_info_frame()
        self.__create_solver_frame()

        self.__check_game_running()
        tk.mainloop()

    def __check_game_running(self):
        try:
            self.__menu = Navigator()
            self.__update_status(True)
            self.__game_var.set(self.__menu.title)

        except RuntimeError: # Make custom error
            self.__update_status(False)
            self.__menu = None
            self.__game_var.set(-1)

    def __update_status(self, status):
        self.__game_running = status
        self.__check_ready_to_solve()

        state = tk.NORMAL if status else tk.DISABLED
        self.__save_optionmenu.configure(state=state)
        self.__set_optionmenu.configure(state=state)
        self.__level_optionmenu.configure(state=state)

    def __load_game(self):
        try:
            self.__menu.close_game()
        except:
            pass

        self.__update_status(False)

        self.__root.update()
        Popen([r'C:\Program Files (x86)\Steam\steam.exe',
               r'steam://rungameid/'+GAMEIDS[self.__game_var.get()]],
              shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

        while True:
            try:
                self.__menu = Navigator()
                if self.__menu.title == self.__game_var.get():
                    self.__update_status(True)
                    break
                else:
                    self.__update_status(False)

            except RuntimeError:
                self.__update_status(False)

        self.__menu.wait_until_loaded()

    def __on_game_var_update(self, *args):
        state = tk.NORMAL if self.__game_var.get() == 'Hexcells Infinite' else tk.DISABLED
        self.__generator_radiobutton.configure(state=state)

    def __create_info_frame(self):
        self.__info_frame = tk.Frame(self.__root)

        self.__info_label = tk.Label(self.__info_frame,
                                     font=('Arial Black', 10),
                                     text='-'*70+'\n'+'Game Options\n'+'-'*70)

        self.__game_var = tk.StringVar(self.__root)
        self.__game_var.set(-1)
        self.__game_var.trace('w', self.__on_game_var_update)

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

        save_slots = ['1', '2', '3']
        self.__save_var = tk.StringVar(self.__root)
        self.__save_var.set('1')
        self.__save_var.trace('w', self.__on_save_change)

        self.__save_frame = tk.Frame(self.__info_frame)
        self.__save_label = tk.Label(self.__save_frame,
                                    font=('Arial', 9),
                                    text='Select Save:')

        self.__save_optionmenu = tk.OptionMenu(self.__save_frame,
                                               self.__save_var,
                                               *save_slots)
        self.__save_label.pack(side='left')
        self.__save_optionmenu.pack()

        self.__info_label.grid(sticky='w', row=0, column=0, columnspan=2)
        self.__game_1_radiobutton.grid(sticky='w', row=1, column=0)
        self.__game_2_radiobutton.grid(sticky='w', row=2, column=0)
        self.__game_3_radiobutton.grid(sticky='w', row=3, column=0)

        self.__save_frame.grid(sticky='w', row=1, column=1)

        self.__info_frame.grid(row=0, column=0)

    def __create_solver_frame(self):
        self.__solver_frame = tk.Frame(self.__root)

        self.__solve_var = tk.IntVar(self.__root)
        self.__solve_var.set(0)
        self.__solve_var.trace('w', self.__solver_radiobutton_callback)

        self.__solve_label = tk.Label(self.__solver_frame,
                                      font=('Arial Black', 10),
                                      text='-'*70+'\n'+'Solver Options\n'+'-'*70)

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

        sets = ['1', '2', '3', '4', '5', '6']
        self.__set_var = tk.StringVar(self.__root)
        self.__set_var.set('-')
        self.__set_var.trace('w', self.__on_set_change)

        self.__set_frame = tk.Frame(self.__solver_frame)
        self.__set_label = tk.Label(self.__set_frame,
                                    font=('Arial', 9),
                                    text='Select Set:')

        self.__set_optionmenu = tk.OptionMenu(self.__set_frame,
                                              self.__set_var,
                                              *sets)
        self.__set_label.pack(side='left')
        self.__set_optionmenu.pack()

        levels = ['1', '2', '3', '4', '5', '6']
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
            if self.__game_running:
                self.__set_optionmenu.configure(state=tk.NORMAL)
                self.__level_optionmenu.configure(state=tk.NORMAL)

        elif self.__solve_var.get() == 1:
            self.__level_var.set('-')
            if self.__game_running:
                self.__set_optionmenu.configure(state=tk.NORMAL)
            self.__level_optionmenu.configure(state=tk.DISABLED)

        else:
            self.__set_var.set('-')
            self.__level_var.set('-')
            self.__set_optionmenu.configure(state=tk.DISABLED)
            self.__level_optionmenu.configure(state=tk.DISABLED)

        self.__check_ready_to_solve()

    def __on_save_change(self, *args):
        pass

    def __on_set_change(self, *args):
        levels = ['1', '2', '3', '4', '5', '6']
        if self.__game_var.get() == 'Hexcells':
            if self.__set_var.get() != '1':
                levels.pop()
            if self.__set_var.get() == '2':
                levels.pop()

        self.__level_optionmenu['menu'].delete(0, 'end')
        for level in levels:
            self.__level_optionmenu['menu'].add_command(label=level, command=tk._setit(self.__level_var, level))

        self.__level_var.set('-')
        self.__check_ready_to_solve()

    def __check_ready_to_solve(self, *args):
        if ((not self.__game_running) or
            (self.__solve_var.get() == 0 and (self.__set_var.get() == '-' or self.__level_var.get() == '-')) or
            (self.__solve_var.get() == 1 and self.__set_var.get() == '-')):
            self.__solve_button.configure(state=tk.DISABLED)
        else:
            self.__solve_button.configure(state=tk.NORMAL)

    def __solve(self):
        if self.__solve_var.get() == 0:
            self.__menu.solve_level(self.__save_var.get(), self.__set_var.get()+'-'+self.__level_var.get())
        elif self.__solve_var.get() == 1:
            self.__menu.solve_set(self.__save_var.get(), self.__set_var.get())
        elif self.__solve_var.get() == 2:
            self.__menu.solve_game(self.__save_var.get())
        elif self.__solve_var.get() == 3:
            self.__menu.level_generator()

if __name__ == '__main__':
    application = GUI()
