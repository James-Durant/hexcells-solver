import os

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

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
        # self.__window_width = 300
        # self.__window_height = 600
        # self.__root.geometry('{0}x{1}+0+0'.format(self.__window_width, self.__window_height))

        self.__create_info_frame()
        self.__create_solver_frame()
        self.__create_learning_frame()

        self.__check_game_running()
        tk.mainloop()

    def __check_game_running(self):
        try:
            self.__menu = Navigator()
            self.__update_status(True)
            self.__game_var.set(self.__menu.title)

        except RuntimeError:  # Make custom error
            self.__update_status(False)
            self.__menu = None
            self.__game_var.set('')

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
               r'steam://rungameid/' + GAMEIDS[self.__game_var.get()]],
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
                                     text='-' * 70 + '\n' + 'Game Options\n' + '-' * 70)

        self.__game_var = tk.StringVar(self.__root)
        self.__game_var.set('')
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
                                      text='-' * 70 + '\n' + 'Solver Options\n' + '-' * 70)

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
                                        state='disabled',
                                        font=('Arial', 10),
                                        command=self.__solve)

        self.__solve_button.grid(sticky='nesw', row=5, column=0, columnspan=2, pady=(10, 0))

        self.__solver_frame.grid(row=1, column=0)

    def __solver_radiobutton_callback(self, *args):
        if self.__solve_var.get() == 0:
            if self.__game_running:
                self.__set_optionmenu.configure(state='normal')
                self.__level_optionmenu.configure(state='normal')

        elif self.__solve_var.get() == 1:
            self.__level_var.set('-')
            if self.__game_running:
                self.__set_optionmenu.configure(state='normal')
            self.__level_optionmenu.configure(state='disabled')

        else:
            self.__set_var.set('-')
            self.__level_var.set('-')
            self.__set_optionmenu.configure(state='disabled')
            self.__level_optionmenu.configure(state='disabled')

        self.__check_ready_to_solve()

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

    def __check_ready_to_solve(self):
        if ((not self.__game_running) or
                (self.__solve_var.get() == 0 and (self.__set_var.get() == '-' or self.__level_var.get() == '-')) or
                (self.__solve_var.get() == 1 and self.__set_var.get() == '-')):
            self.__solve_button.configure(state='disabled')
        else:
            self.__solve_button.configure(state='normal')

    def __solve(self):
        if self.__solve_var.get() == 0:
            self.__menu.solve_level(self.__save_var.get(), self.__set_var.get() + '-' + self.__level_var.get())
        elif self.__solve_var.get() == 1:
            self.__menu.solve_set(self.__save_var.get(), self.__set_var.get())
        elif self.__solve_var.get() == 2:
            self.__menu.solve_game(self.__save_var.get())
        elif self.__solve_var.get() == 3:
            self.__menu.level_generator()

    def __create_learning_frame(self):
        self.__learning_frame = tk.Frame(self.__root)

        self.__model_var = tk.IntVar(self.__root)
        self.__model_var.set(0)
        self.__model_var.trace('w', self.__on_model_var_update)

        self.__learning_label = tk.Label(self.__learning_frame,
                                         font=('Arial Black', 10),
                                         text='-' * 70 + '\n' + 'Learning Options\n' + '-' * 70)

        self.__new_radiobutton = tk.Radiobutton(self.__learning_frame,
                                                variable=self.__model_var,
                                                value=0,
                                                text='New Model',
                                                font=('Arial Black', 9))

        self.__load_radiobutton = tk.Radiobutton(self.__learning_frame,
                                                 variable=self.__model_var,
                                                 value=1,
                                                 text='Load Model',
                                                 font=('Arial Black', 9))

        modes = ['Online', 'Offline']
        self.__mode_var = tk.StringVar(self.__root)
        self.__mode_var.set('Online')

        self.__mode_frame = tk.Frame(self.__learning_frame)
        self.__mode_label = tk.Label(self.__mode_frame,
                                     font=('Arial', 9),
                                     text='Select Mode:')

        self.__mode_optionmenu = tk.OptionMenu(self.__mode_frame,
                                               self.__mode_var,
                                               *modes)
        self.__mode_label.pack(side='left')
        self.__mode_optionmenu.pack()

        self.__test_var = tk.BooleanVar(self.__root)
        self.__test_var.set(False)

        self.__test_checkbutton = tk.Checkbutton(self.__learning_frame,
                                                 text='Test Only?',
                                                 variable=self.__test_var,
                                                 onvalue=True,
                                                 offvalue=False)

        self.__mode_frame.grid(sticky='w', row=1, column=1)
        self.__test_checkbutton.grid(sticky='w', row=2, column=1)

        self.__learning_label.grid(sticky='w', row=0, column=0, columnspan=2)
        self.__new_radiobutton.grid(sticky='w', row=1, column=0)
        self.__load_radiobutton.grid(sticky='w', row=2, column=0)

        self.__epochs_frame = tk.Frame(self.__learning_frame)
        self.__epochs_label = tk.Label(self.__epochs_frame,
                                       text='Epochs: ')

        self.__epochs_var = tk.StringVar()
        self.__epochs_var.set('1')
        self.__epochs_entry = tk.Entry(self.__epochs_frame,
                                       textvariable=self.__epochs_var)

        self.__epochs_label.pack(side='left')
        self.__epochs_entry.pack(expand=True, fill='both')
        self.__epochs_frame.grid(sticky='nesw', row=4, column=0, columnspan=2)

        self.__batch_size_frame = tk.Frame(self.__learning_frame)
        self.__batch_size_label = tk.Label(self.__batch_size_frame,
                                           text='Batch Size: ')

        self.__batch_size_var = tk.StringVar()
        self.__batch_size_var.set('64')
        self.__batch_size_entry = tk.Entry(self.__batch_size_frame,
                                           textvariable=self.__batch_size_var)

        self.__batch_size_label.pack(side='left')
        self.__batch_size_entry.pack(expand=True, fill='both')
        self.__batch_size_frame.grid(sticky='nesw', row=5, column=0, columnspan=2)

        self.__learning_rate_frame = tk.Frame(self.__learning_frame)
        self.__learning_rate_label = tk.Label(self.__learning_rate_frame,
                                              text='Learning Rate: ')

        self.__learning_rate_var = tk.StringVar()
        self.__learning_rate_var.set('0.01')
        self.__learning_rate_entry = tk.Entry(self.__learning_rate_frame,
                                              textvariable=self.__learning_rate_var)

        self.__learning_rate_label.pack(side='left')
        self.__learning_rate_entry.pack(expand=True, fill='both')
        self.__learning_rate_frame.grid(sticky='nesw', row=6, column=0, columnspan=2)

        self.__discount_rate_frame = tk.Frame(self.__learning_frame)
        self.__discount_rate_label = tk.Label(self.__discount_rate_frame,
                                              text='Discount Rate: ')

        self.__discount_rate_var = tk.StringVar()
        self.__discount_rate_var.set('0.05')
        self.__discount_rate_entry = tk.Entry(self.__discount_rate_frame,
                                              textvariable=self.__discount_rate_var)

        self.__discount_rate_label.pack(side='left')
        self.__discount_rate_entry.pack(expand=True, fill='both')
        self.__discount_rate_frame.grid(sticky='nesw', row=7, column=0, columnspan=2)

        self.__exploration_rate_frame = tk.Frame(self.__learning_frame)
        self.__exploration_rate_label = tk.Label(self.__exploration_rate_frame,
                                                 text='Exploration Rate: ')

        self.__exploration_rate_var = tk.StringVar()
        self.__exploration_rate_var.set('0.95')
        self.__exploration_rate_entry = tk.Entry(self.__exploration_rate_frame,
                                                 textvariable=self.__exploration_rate_var)

        self.__exploration_rate_label.pack(side='left')
        self.__exploration_rate_entry.pack(expand=True, fill='both')
        self.__exploration_rate_frame.grid(sticky='nesw', row=8, column=0, columnspan=2)

        self.__path_frame = tk.Frame(self.__learning_frame)
        self.__path_button = tk.Button(self.__path_frame,
                                       text='Select Model: ',
                                       state='disabled',
                                       command=self.__select_model_path)

        self.__model_path_var = tk.StringVar()
        self.__model_path_var.set('')
        self.__path_entry = tk.Entry(self.__path_frame,
                                     state='disabled',
                                     textvariable=self.__model_path_var)

        self.__path_button.pack(side='left')
        self.__path_entry.pack(expand=True, fill='both')
        self.__path_frame.grid(sticky='nesw', row=9, column=0, columnspan=2, pady=10)

        self.__train_button = tk.Button(self.__learning_frame,
                                        text='Run',
                                        font=('Arial', 10),
                                        command=self.__train)

        self.__train_button.grid(sticky='nesw', row=10, column=0, columnspan=2, pady=(10, 0))

        self.__learning_frame.grid(row=2, column=0)

    def __on_model_var_update(self, *args):
        if self.__model_var.get() == 0:
            self.__path_button.configure(state='disabled')
            self.__path_entry.configure(state='disabled')
            self.__model_path_var.set('')
        elif self.__model_var.get() == 1:
            self.__path_button.configure(state='normal')
            self.__path_entry.configure(state='normal')

    def __train(self):
        try:
            epochs = int(self.__epochs_var.get())
            batch_size = int(self.__batch_size_var.get())
            learning_rate = float(self.__learning_rate_var.get())
            discount_rate = float(self.__discount_rate_var.get())
            exploration_rate = float(self.__exploration_rate_var.get())
            model_path = self.__model_path_var.get()
            model_path = None if model_path == '' else model_path
            
            assert epochs > 0
            assert batch_size > 0
            assert learning_rate > 0
            assert 0 <= discount_rate <= 1
            assert 0 <= exploration_rate <= 1

        except (ValueError, AssertionError):
            messagebox.showerror('Error', 'Invalid hyperparameter value(s) given')
            return

        from learn import Trainer

        try:
            if self.__mode_var.get() == 'Offline':
                Trainer.train_offline(test_only=self.__mode_var.get(),
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      learning_rate=learning_rate,
                                      discount_rate=discount_rate,
                                      exploration_rate=exploration_rate,
                                      model_path=model_path)

            elif self.__mode_var.get() == 'Online':
                raise NotImplementedError

        except (FileNotFoundError, IOError):
            messagebox.showerror('Error', 'Invalid model path given')

    def __select_model_path(self):
        file_path = os.path.dirname(os.path.realpath(__file__))
        models_dir = os.path.join(file_path, 'resources', 'models')
        model_path = filedialog.askopenfilename(parent=self.__root, initialdir=models_dir, title='Model Selection',
                                                filetypes=[('HDF5 file', '.h5')])
        self.__model_path_var.set(model_path)


if __name__ == '__main__':
    application = GUI()
