import os
import random

def state_game(num_states):
    names = os.listdir(f'states/img_env/training/')
    num_names = len(names)
    correct = 0
    incorrect = 0
    for i in range(num_states):
        description = "This is a temporary description"
        name_idx = random.randint(0, num_names)
        print_state(names[name_idx])
        print(description)
        inp = input("Is this prompt correct? (0=no, 1=yes): ")
        if inp == "0":
            incorrect += 1
        elif inp == "1":
            correct += 1
        else:
            print("Input in wrong range, continuing...")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Hit rate: {(correct / (correct + incorrect)):.2f}")
        
def print_state(state_str):
    state_list = state_str.split("_")
    grid = state_list[0]
    agent = state_list[1]
    grid = grid.replace("H", "O")
    grid = grid.replace("F", " ")
    grid = list(grid)
    grid[int(agent)] = "A"
    grid = "".join(grid)
    print(state_str)
    print("-"*17)
    print(f"| {grid[0]} | {grid[1]} | {grid[2]} | {grid[3]} |")
    print("-"*17)
    print(f"| {grid[4]} | {grid[5]} | {grid[6]} | {grid[7]} |")
    print("-"*17)
    print(f"| {grid[8]} | {grid[9]} | {grid[10]} | {grid[11]} |")
    print("-"*17)
    print(f"| {grid[12]} | {grid[13]} | {grid[14]} | {grid[15]} |")
    print("-"*17)
    
    
if __name__ == '__main__':
    state_game(100)