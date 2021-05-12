from navigate import Navigator
            
if __name__ == '__main__':
    menu = Navigator()
    menu.load_save_slot(2)
    
    menu.solve_level('4-1')
    #menu.solve_world('1')
    #menu.solve_game()