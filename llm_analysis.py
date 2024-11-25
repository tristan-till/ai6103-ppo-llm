import os
import random

import utils.render_env as render_env
import utils.enums as enums
import utils.req as req

def state_game(num_states, mode=enums.EnvMode.TRAIN):
    names = os.listdir(f'states/img_env/training/')
    num_names = len(names)
    correct = 0
    incorrect = 0
    for i in range(num_states):
        name_idx = random.randint(0, num_names)
        state_str = names[name_idx].split(".")[0]
        state_list = state_str.split("_")
        grid = state_list[0]
        agent = state_list[1]
        dir = state_list[2]
        print_state(grid, agent)
        # description = render_env.get_llava(grid, f"{agent}_{dir}", mode)
        # prompt = render_env.get_custom_state_str(grid, agent)
        prompt = render_env.get_llava(grid, f"{agent}_{dir}", mode)
        description = req.get_llama(prompt)
        print(f"Prompt: {prompt}")
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
        
def print_state(grid, agent):
    grid = grid.replace("H", "O")
    grid = grid.replace("F", " ")
    grid = list(grid)
    grid[int(agent)] = "A"
    grid = "".join(grid)
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
    state_game(10, enums.EnvMode.TRAIN)