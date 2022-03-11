import os
import sys

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from subprocess import Popen

from data import GAMEIDS
from parse import STEAM_PATH
from navigate import Navigator
from window import WindowNotFoundError
from learn import Trainer, LEARNING_RATE, DISCOUNT_RATE, EXPLORATION_RATE


class GUI:
    """Contains the code for the GUI and its associated functionality."""

    def __init__(self):
        # Create the main GUI window
        self.__root = tk.Tk()
        self.__root.title('HexSolver')
        self.__root.resizable(width=False, height=False)

        self.__button_font = ('Arial', 10)
        self.__title_font = ('Arial Black', 10)
        self.__radiobutton_font = ('Arial Black', 9)

        # Add the three main components of the GUI.
        self.__setup_vars()
        self.__create_game_frame()
        self.__create_solver_frame()
        self.__create_learning_frame()

        # Run the GUI.
        self.__check_game_running()
        tk.mainloop()

    def __setup_vars(self):
        """Define the variables set by the components of the GUI."""
        # Which game in the Hexcells series is currently running.
        self.__game_var = tk.StringVar(self.__root)

        # Which save slot to load.
        self.__save_var = tk.StringVar(self.__root)
        self.__save_var.set('1') # Save slot 1 by default.

        # Whether to add a delay after clicking cells (for the particle effect to clear).
        self.__delay_var = tk.BooleanVar(self.__root)
        self.__delay_var.set(False) # No delay by default.

        # Whether to solve a specific level, a set of levels, an entire game
        # or a randomly generate level.
        self.__solve_var = tk.IntVar(self.__root)
        self.__solve_var.set(0) # Initially a specific level.
        self.__solve_var.trace('w', self.__solver_radiobutton_callback)

        # Which level set to use for solving.
        self.__set_var = tk.StringVar(self.__root)
        self.__set_var.set('-') # Not assigned initially.
        self.__set_var.trace('w', self.__on_set_change)

        # Which level within a set to use for solving.
        self.__level_var = tk.StringVar(self.__root)
        self.__level_var.set('-') # Not assigned initially.
        self.__level_var.trace('w', self.__check_ready_to_solve)

        # Whether to create a new model or load an existing one.
        self.__model_var = tk.IntVar(self.__root)
        self.__model_var.set(0)
        self.__model_var.trace('w', self.__on_model_var_update)

        # Whether to train a model when running it or not.
        self.__train_var = tk.BooleanVar(self.__root)
        self.__train_var.set(False)

        # Whether to use experience replay or not.
        self.__replay_var = tk.BooleanVar(self.__root)
        self.__replay_var.set(False)

        # Whether to use double deep Q-learning or not.
        self.__double_var = tk.BooleanVar(self.__root)
        self.__double_var.set(False)

        # Number of epochs to run a model for.
        self.__epochs_var = tk.StringVar()
        self.__epochs_var.set('3')

        # Value for the learning rate hyperparameter.
        self.__learning_rate_var = tk.StringVar()
        self.__learning_rate_var.set(str(LEARNING_RATE))

        # Value for the discount rate hyperparameter.
        self.__discount_rate_var = tk.StringVar()
        self.__discount_rate_var.set(str(DISCOUNT_RATE))

         # Value for the exploration rate hyperparameter.
        self.__exploration_rate_var = tk.StringVar()
        self.__exploration_rate_var.set(str(EXPLORATION_RATE))

        # File path to a model to load.
        self.__model_path_var = tk.StringVar()
        self.__model_path_var.set('')

        # Whether to use online or offline learning.
        self.__mode_var = tk.StringVar(self.__root)
        self.__mode_var.set('Online')
        self.__mode_var.trace('w', self.__on_mode_var_update)

    def __create_game_frame(self):
        """Adds the components related to game options to the GUI."""
        # Create a frame to contain the components for game options.
        self.__game_frame = tk.Frame(self.__root)
        self.__game_frame.grid(row=0, column=0)

        # Create a title for the frame.
        frame_title = '-' * 70 + '\n' + 'Game Options\n' + '-' * 70
        self.__game_label = tk.Label(self.__game_frame, text=frame_title, font=self.__title_font)
        self.__game_label.grid(sticky='w', row=0, column=0, columnspan=2)

        # Create the three radiobuttons for selecting the game to load.
        self.__game_radiobuttons = []
        for i, game in enumerate(GAMEIDS, 1):
            radiobutton = tk.Radiobutton(self.__game_frame, variable=self.__game_var, value=game,
                                         text=game, font=self.__radiobutton_font, command=self.__load_game)
            radiobutton.grid(sticky='w', row=i, column=0)
            self.__game_radiobuttons.append(radiobutton)

        # Create a frame to contain the "Select Save" text and a dropdown menu.
        saves = [1, 2, 3]
        self.__save_frame = tk.Frame(self.__game_frame)
        self.__save_label = tk.Label(self.__save_frame, text='Select Save:')
        self.__save_optionmenu = tk.OptionMenu(self.__save_frame, self.__save_var, *saves)
        self.__save_label.pack(side='left')
        self.__save_optionmenu.pack()
        self.__save_frame.grid(sticky='w', row=1, column=1)

        # Create a checkbox to enable or disable a delay after clicking cells.
        self.__delay_checkbutton = tk.Checkbutton(self.__game_frame, text='Particle Effect', variable=self.__delay_var)
        self.__delay_checkbutton.grid(sticky='w', row=2, column=1)

    def __create_solver_frame(self):
        """Adds the components relating to solver options to the GUI."""
        # Create a frame to contain the components for solver options.
        self.__solver_frame = tk.Frame(self.__root)
        self.__solver_frame.grid(row=1, column=0)

        # Create a title for the frame.
        frame_title = '-' * 70 + '\n' + 'Solver Options\n' + '-' * 70
        self.__solve_label = tk.Label(self.__solver_frame, text=frame_title, font=self.__title_font)
        self.__solve_label.grid(sticky='w', row=0, column=0, columnspan=2)

        # Create a radiobutton for each of the solving options.
        self.__solver_radiobuttons = []
        labels = ['Specific Level', 'Specific Set', 'Entire Game', 'Level Generator']
        for i, label in enumerate(labels):
            radiobutton = tk.Radiobutton(self.__solver_frame, variable=self.__solve_var,
                                         value=i, text=label, font=self.__radiobutton_font)
            radiobutton.grid(sticky='w', row=i+1, column=0)
            self.__solver_radiobuttons.append(radiobutton)

        # Create a frames containing a set/level selection dropdowns with an associated label.
        self.__solver_optionmenus = []
        labels = ['Select Set', 'Select Level']
        variables = [self.__set_var, self.__level_var]
        for i, (variable, label) in enumerate(zip(variables, labels), 1):
            options = ['1', '2', '3', '4', '5', '6']
            frame = tk.Frame(self.__solver_frame)
            label = tk.Label(frame, text=f'{label}:')
            optionmenu = tk.OptionMenu(frame, variable, *options)
            label.pack(side='left')
            optionmenu.pack()
            frame.grid(sticky='w', row=i, column=1)
            self.__solver_optionmenus.append(optionmenu)

        # Create a button that can be clicked to run the solver on the chosen level(s).
        self.__solve_button = tk.Button(self.__solver_frame, text='Solve', font=self.__button_font,
                                        state=tk.DISABLED, command=self.__run_solver)
        self.__solve_button.grid(sticky='nesw', row=5, column=0, columnspan=2, pady=(10, 0))

    def __create_learning_frame(self):
        """Adds the components related to machine learning options to the GUI."""
        # Create a frame to contain the components for game options.
        self.__learning_frame = tk.Frame(self.__root)
        self.__learning_frame.grid(row=2, column=0)

        # Create a title for the frame.
        frame_title = '-' * 70 + '\n' + 'Learning Options\n' + '-' * 70
        self.__learning_label = tk.Label(self.__learning_frame, text=frame_title, font=self.__title_font)
        self.__learning_label.grid(sticky='w', row=0, column=0, columnspan=2)

        # Create radiobuttons for creating a new model or loading an existing one.
        self.__learning_radiobuttons = []
        for i, label in enumerate(['New Model', 'Load Model']):
            radiobutton = tk.Radiobutton(self.__learning_frame, variable=self.__model_var,
                                         value=i, text=label, font=self.__radiobutton_font)
            radiobutton.grid(sticky='w', row=i+1, column=0)
            self.__learning_radiobuttons.append(radiobutton)

        # Create checkbuttons for setting whether a model should be trained, and whether
        # experience replay or double deep Q-learning should be used.
        self.__learning_checkbuttons = []
        variables = [self.__train_var, self.__replay_var, self.__double_var]
        labels = ['Train', 'Experience Replay', 'Double DQN']
        for i, (variable, label) in enumerate(zip(variables, labels), 1):
            checkbutton = tk.Checkbutton(self.__learning_frame, text=label, variable=variable)
            checkbutton.grid(sticky='w', row=i, column=1)
            self.__learning_checkbuttons.append(checkbutton)

        # Create entry boxes (with an associated label) for each of the model hyperparameters.
        self.__learning_entries = []
        variables = [self.__epochs_var, self.__learning_rate_var, self.__discount_rate_var, self.__exploration_rate_var]
        labels = ['Epochs', 'Learning Rate', 'Discount Rate', 'Exploration Rate']
        for i, (variable, label) in enumerate(zip(variables, labels), 4):
            frame = tk.Frame(self.__learning_frame)
            label = tk.Label(frame,  text=f'{label}: ')
            entry = tk.Entry(frame, textvariable=variable)
            label.pack(side='left')
            entry.pack(expand=True, fill='both')
            frame.grid(sticky='nesw', row=i, column=0, columnspan=2)
            self.__learning_entries.append(entry)

        # Create a button that brings up a file selection dialogue box when clicked.
        # Also create an entry to display the selected file.
        self.__path_frame = tk.Frame(self.__learning_frame)
        self.__path_button = tk.Button(self.__path_frame, text='Select Model: ',
                                       state=tk.DISABLED, command=self.__select_model_path)
        self.__path_entry = tk.Entry(self.__path_frame, state=tk.DISABLED, textvariable=self.__model_path_var)
        self.__path_button.pack(side='left')
        self.__path_entry.pack(expand=True, fill='both')
        self.__path_frame.grid(sticky='nesw', row=8, column=0, columnspan=2, pady=10)

        # Create a dropdown menu for selecting between online and offline learning.
        modes = ['Online', 'Offline']
        self.__run_frame = tk.Frame(self.__learning_frame)
        self.__run_frame.grid(sticky='nesw', row=9, column=0, columnspan=2, pady=(10, 0))
        self.__mode_optionmenu = tk.OptionMenu(self.__run_frame, self.__mode_var, *modes)
        self.__mode_optionmenu.pack(side='right')

        # Create a button that, when clicked, runs a chosen model.
        self.__train_button = tk.Button(self.__run_frame, text='Run', font=self.__button_font,
                                        state=tk.DISABLED, command=self.__run_model)
        self.__train_button.pack(expand=True, fill='x')

    def __check_game_running(self):
        """Check whether a Hexcells game is currently running."""
        try:
            # Try to capture a game window and update the GUI accordingly.
            self.__menu = Navigator()
            self.__game_running = True
            self.__game_var.set(self.__menu.title)
            self.__update_status(True)

        except KeyboardInterrupt:
            # Exit on Control + C event.
            sys.exit()

        except WindowNotFoundError:
            # Update the GUI if no window can be found.
            self.__update_status(False)
            self.__menu = None
            self.__game_var.set('None')

        except RuntimeError as e:
            # Display the error if a runtime error is raised.
            messagebox.showerror('Error', str(e))
            sys.exit()

    def __update_status(self, status):
        """Update the components of the GUI based on whether a game window is open or not.

        Args:
            status (bool): whether a game window is active or not.
        """
        # Check if the solve button can be enabled.
        self.__game_running = status
        self.__check_ready_to_solve()

        # Disable the save, set and level dropdowns if no game is running.
        state = tk.NORMAL if status else tk.DISABLED
        self.__save_optionmenu.configure(state=state)
        self.__solver_optionmenus[0].configure(state=state)
        self.__solver_optionmenus[1].configure(state=state)

        # Only enable the option to run the solver on the level generator, or run a model, if the
        # game is Hexcells Infinite (both require level generator which is not in the other games).
        state = tk.NORMAL if status and self.__game_var.get() == 'Hexcells Infinite' else tk.DISABLED
        self.__solver_radiobuttons[3].configure(state=state)
        self.__train_button.configure(state=state)

    def __load_game(self):
        """Loads a chosen game in the Hexcells series."""
        try:
            # Try to close the game currently open (if applicable).
            self.__menu.close_game()

        except KeyboardInterrupt:
            # Exit on Control + C event.
            sys.exit()

        except Exception:
            pass # Ignore any other exceptions.

        # Now that there are no games open, disable the relevant GUI components.
        self.__update_status(False)

        # Load the chosen game by sending a command to the Steam executable.
        self.__root.update()
        steam_path = os.path.join(STEAM_PATH, 'steam.exe')
        Popen([steam_path, f'steam://rungameid/{GAMEIDS[self.__game_var.get()]}'],
              shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

        # Try to capture the newly opened game window.
        while True:
            try:
                # Check that the window corresponds to the correct game.
                self.__menu = Navigator()
                if self.__menu.title == self.__game_var.get():
                    self.__update_status(True)
                    break
                else:
                    self.__update_status(False)

            except KeyboardInterrupt:
                # Exit on Control + C event.
                sys.exit()

            except WindowNotFoundError:
                # There is no window yet.
                self.__update_status(False)

        # Wait until the main menu screen has fully loaded before continuing.
        self.__menu.wait_until_loaded()

    def __solver_radiobutton_callback(self, *args):
        """This method is called when the solving option is changed."""
        # Get the level and set selection dropdowns.
        level_optionmenu, set_optionmenu = self.__solver_optionmenus

        # Check if the solver is to be run on a specific level.
        if self.__solve_var.get() == 0:
            # If the game is running, enable the set and level selection dropdowns.
            if self.__game_running:
                level_optionmenu.configure(state=tk.NORMAL)
                set_optionmenu.configure(state=tk.NORMAL)

        # Check if the solver is to be run on a set of levels.
        elif self.__solve_var.get() == 1:
            self.__level_var.set('-') # Reset the level selection dropdown.
            # If the game is running, only enable the set selection dropdown.
            if self.__game_running:
                level_optionmenu.configure(state=tk.NORMAL)
            set_optionmenu.configure(state=tk.DISABLED)

        else:
            # Otherwise, reset and disable the set and level selection dropdowns.
            self.__set_var.set('-')
            self.__level_var.set('-')
            level_optionmenu.configure(state=tk.DISABLED)
            set_optionmenu.configure(state=tk.DISABLED)

        # Check if the solve button can now be enabled.
        self.__check_ready_to_solve()

    def __on_set_change(self, *args):
        """This method is called when the set selection variable is updated."""
        # If the selected game is Hexcells, update the options in the level selection dropdown.
        levels = ['1', '2', '3', '4', '5', '6']
        if self.__game_var.get() == 'Hexcells':
            # Only set 1 has 6 levels in Hexcells.
            if self.__set_var.get() != '1':
                levels.pop()
            # Set 2 only has 4 levels so remove another.
            if self.__set_var.get() == '2':
                levels.pop()

        # Update the level selection dropdown with these new changes.
        level_optionmenu = self.__solver_optionmenus[1]
        level_optionmenu['menu'].delete(0, 'end')
        for level in levels:
            level_optionmenu['menu'].add_command(label=level, command=tk._setit(self.__level_var, level))

        # Reset the level selection variable and check if the solve button can be enabled.
        self.__level_var.set('-')
        self.__check_ready_to_solve()

    def __check_ready_to_solve(self, *args):
        """Check if the solve button can be enabled."""
        # Check if the game is running and whether a valid set-level selection has been made.
        if ((not self.__game_running) or
            (self.__solve_var.get() == 0 and (self.__set_var.get() == '-' or self.__level_var.get() == '-')) or
            (self.__solve_var.get() == 1 and self.__set_var.get() == '-')):
            self.__solve_button.configure(state=tk.DISABLED)
        else:
            self.__solve_button.configure(state=tk.NORMAL)

    def __on_model_var_update(self, *args):
        """This method is called when the model variable is changed."""
        # If a new model is to be created, disable the file path entry for loading a model.
        if self.__model_var.get() == 0:
            self.__path_button.configure(state=tk.DISABLED)
            self.__path_entry.configure(state=tk.DISABLED)
            self.__model_path_var.set('')

        # Otherwise, enable the model path entry and button.
        elif self.__model_var.get() == 1:
            self.__path_button.configure(state=tk.NORMAL)
            self.__path_entry.configure(state=tk.NORMAL)

    def __on_mode_var_update(self, *args):
        """This method is called when the learning mode variable is changed."""
        # If using online learning, check that the game is running.
        if self.__mode_var.get() == 'Online':
            self.__check_game_running()

        # If using offline learning, a game window does not need to be open.
        elif self.__mode_var.get() == 'Offline':
            self.__train_button.configure(state=tk.NORMAL)

    def __select_model_path(self):
        """This method is called when the file selection button is clicked."""
        # Get the path to resources/models using the path of this file.
        models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'models')

        # Open a file selection dialogue box for the user to select a HDF5 file.
        model_path = filedialog.askopenfilename(parent=self.__root, initialdir=models_dir,
                                                title='Model Selection', filetypes=[('HDF5 file', '.h5')])
        self.__model_path_var.set(model_path)

    def __run_solver(self):
        """This method is called when the solve button is clicked."""
        # Check if the game is running. If not, display an error message.
        self.__check_game_running()
        if not self.__game_running:
            messagebox.showerror('Error', 'Hexcells window not found')
            return

        try:
            # Solve a specific level given by the user via the dropdown menus.
            if self.__solve_var.get() == 0:
                level = f'{self.__set_var.get()}-{self.__level_var.get()}'
                self.__menu.solve_level(self.__save_var.get(), level, self.__delay_var.get())

            # Solve a specific set of levels given by the user via the dropdown menu.
            elif self.__solve_var.get() == 1:
                self.__menu.solve_set(self.__save_var.get(), self.__set_var.get(), self.__delay_var.get())

            # Solve an entire game.
            elif self.__solve_var.get() == 2:
                self.__menu.solve_game(self.__save_var.get(), self.__delay_var.get())

            # Solve a randomly generate level.
            elif self.__solve_var.get() == 3:
                self.__menu.level_generator(1, self.__delay_var.get())

        except RuntimeError as e:
            # Display any error messages that arise during solving.
            messagebox.showerror('Error', str(e))

    def __run_model(self):
        """This method is called when the run button is clicked."""
        try:
            # Try to convert each hyperparameter entry to its numerical datatype.
            epochs = int(self.__epochs_var.get())
            learning_rate = float(self.__learning_rate_var.get())
            discount_rate = float(self.__discount_rate_var.get())
            exploration_rate = float(self.__exploration_rate_var.get())
            model_path = self.__model_path_var.get()
            model_path = None if model_path == '' else model_path

            # Check that the given values are valid.
            assert epochs > 0
            assert learning_rate > 0
            assert 0 <= discount_rate <= 1
            assert 0 <= exploration_rate <= 1

        except (ValueError, AssertionError):
            # Display an error message if any invalid values are given.
            messagebox.showerror('Error', 'Invalid hyperparameter value given')
            return

        try:
            # Check if offline learning was selected.
            if self.__mode_var.get() == 'Offline':
                Trainer.run_offline(epochs, self.__train_var.get(), learning_rate,
                                    discount_rate, exploration_rate, self.__replay_var.get(),
                                    self.__double_var.get(), model_path=model_path)

            # Check if online learning was selected.
            elif self.__mode_var.get() == 'Online':
                # Make sure that the game is running.
                self.__check_game_running()
                if not self.__game_running:
                    messagebox.showerror('Error', 'Hexcells window not found')
                    return

                # Define a function that can be repeatedly called to run the agent.
                def train(agent):
                    Trainer.run_online(agent, self.__menu.window, self.__delay_var.get(),
                                       self.__train_var.get(), learning_rate, discount_rate,
                                       exploration_rate, self.__replay_var.get(),
                                       self.__double_var.get(), model_path=model_path)

                # Run the agent on the level generator for the given number of epochs.
                self.__menu.level_generator(epochs, self.__delay_var.get(), train)

        except (FileNotFoundError, IOError):
            # Display an error message if an invalid model path was given.
            messagebox.showerror('Error', 'Invalid model path given')


if __name__ == '__main__':
    # Start the GUI if this file is being run.
    GUI()
