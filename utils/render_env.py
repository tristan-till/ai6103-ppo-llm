from PIL import Image
import numpy as np
import io
import base64

from classes.sentence_transformer import SentenceTransformer
import utils.enums as enums
import utils.req as req

border=1
sprite_size=(32, 32)

orientation_map = {
    "0": "left",
    "1": "down",
    "2": "right",
    "3": "up"
}

sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")

def get_coordinates_from_tile(tile):
    tile_x, tile_y = tile
    x = tile_x * (sprite_size[0] + border)
    y = tile_y * (sprite_size[1] + border)
    return x, y

def place_img(sprite_path, background, tile):
    x, y = get_coordinates_from_tile(tile)
    sprite = Image.open(sprite_path).convert("RGBA").resize(sprite_size)
    temp_image = Image.new("RGBA", background.size, (0, 0, 0, 0))
    temp_image.paste(sprite, (x, y), sprite)
    background.alpha_composite(temp_image)
    
def parse_grid(grid):
    start = None
    goal = None
    holes = []
    dim = np.sqrt(len(grid))
    dim = int(dim)
    for i, char in enumerate(grid):
        if char == "S":
            start = (int(i % dim), int(i / dim))
        elif char == "G":
            goal = (int(i % dim), int(i / dim))
        elif char == "H":
            holes.append((int(i % dim), int(i / dim)))
    return dim, start, goal, holes

def generate_background(size, mode):
    if mode == enums.EnvMode.TRAIN:
        sprite_path = "sprites/ice.png"
    else:
        sprite_path = "sprites/black_ice.png"
    grid_width = size * (sprite_size[0] + border) - border
    grid_height = size * (sprite_size[1] + border) - border
    background = Image.new("RGBA", (grid_width, grid_height), (211, 211, 211, 255))
    
    for row in range(size):
        for col in range(size):
            place_img(sprite_path, background, (row, col))
    return background

def place_lakes(background, holes):
    for tile in holes:
        place_img("sprites/hole.png", background, tile)
    return background

def place_stool(background, start):
    place_img("sprites/stool.png", background, start)
    return background
    
def place_goal(background, goal):
    place_img("sprites/goal.png", background, goal)
    return background

def place_agent(background, state_str, size, holes):
    state, orientation = state_str.split("_")
    state = int(state)
    orientation_str = orientation_map[orientation]
    agent_tile = (int(state % size), int(state / size))
    if agent_tile not in holes:
        place_img(f"sprites/elf_{orientation_str}.png", background, agent_tile)
    else:
        place_img(f"sprites/cracked_hole.png", background, agent_tile)
    return background

def render_arr(grid, state_str, mode=enums.EnvMode.TRAIN):
    img = render_state(grid, state_str, mode)
    img = img.convert("RGB")
    img = img.resize((256, 256), Image.NEAREST)
    img = np.array(img, dtype=np.uint8)
    return img

def render_state(grid, state_str, mode):
    assert type(grid) == str, f"Grid must be a string, e.g. 'SHHHFFFFFFHFHFFGHHHHHFFFF', instead got {grid}"
    assert type(state_str) == str, "State must be a string, e.g. '0_0'"
    return _render_state(grid, state_str, mode)

def _render_state(grid, state_str, mode):
    size, start, goal, holes = parse_grid(grid)
    background = generate_background(size, mode)
    background = place_lakes(background, holes)
    background = place_stool(background, start)
    background = place_goal(background, goal)
    background = place_agent(background, state_str, size, holes)
    return background

def base64_from_state_arr(state_arr):
    image = Image.fromarray(state_arr)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

def render_img_and_embedding(grid, state_str, mode):
    render = render_arr(grid, state_str, mode)
    base64_img = base64_from_state_arr(render)
    llava = req.get_llava(base64_img)
    llama = req.get_llama(llava)
    embedding = sentence_transformer.encode(llama)
    state = np.append(render.flatten(), embedding)
    return state

def render_embedding(grid, state_str, mode):
    render = render_arr(grid, state_str, mode)
    base64_img = base64_from_state_arr(render)
    llava = req.get_llava(base64_img)
    llama = req.get_llama(llava)
    embedding = sentence_transformer.encode(llama)
    return embedding
    

if __name__ == '__main__':
    grid = 'SHHHFFFFFFHFHFFGHHHHHFFFF'
    state_str = '0_0'
    background = render_img_and_embedding(grid, state_str, black_ice=True)
    
    
