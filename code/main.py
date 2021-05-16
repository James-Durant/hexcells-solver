from navigate import Navigator
            
if __name__ == '__main__':
    menu = Navigator()
    menu.solve(continuous=True)
    
    #menu.puzzle_generator()
    
    #menu.save_slot(2)
    #menu.solve_level('4-6')
    #menu.solve_world('4')
    #menu.solve_game()
    
    #Hexcells Plus: 5-2 6 misclassified as 5
    #Hexcells Infinite: 1-5 can't solve