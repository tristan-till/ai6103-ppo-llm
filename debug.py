import os
import numpy as np
import time

def main(path):
    names = os.listdir(f'states/llmv2_env/{path}/')
    keys = [file.strip('.npy').split("_") for file in names]
    state_pos = [f"{key[0]}_{key[1]}" for key in keys]
    counter = 0
    state_set = set(state_pos)
    for state in state_set:
        for i in range(4):
            if os.path.isfile(f'states/llmv2_env/{path}/{state}_{i}.npy'):
                arr = np.load(f'states/llmv2_env/{path}/{state}_{i}.npy')
                break
        for i in range(4):
            if not os.path.isfile(f'states/llmv2_env/{path}/{state}_{i}.npy'):
                print(f"saved new {path} state {state}_{i}.npy")
                np.save(f'states/llmv2_env/{path}/{state}_{i}.npy', arr)
                counter += 1

    print(f"generated {counter} new states")
    
if __name__ == '__main__':
    while True:
        main("training")
        main("validation")
        main("testing")
        time.sleep(1)
    
