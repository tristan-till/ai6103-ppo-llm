from PIL import Image

lake_tiles = [(1, 1), (3, 1), (3, 2), (0, 3)]
border=1
sprite_size=(32, 32)
grid_size=4

orientation_map = {
    "0": "left",
    "1": "down",
    "2": "right",
    "3": "up"
}

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

def generate_background(black_ice):
    if not black_ice:
        sprite_path = "sprites/ice.png"
    else:
        sprite_path = "sprites/black_ice.png"
    grid_width = grid_size * (sprite_size[0] + border) - border
    grid_height = grid_size * (sprite_size[1] + border) - border
    background = Image.new("RGBA", (grid_width, grid_height), (211, 211, 211, 255))
    
    for row in range(grid_size):
        for col in range(grid_size):
            place_img(sprite_path, background, (row, col))
    return background

def place_lakes(background):
    for tile in lake_tiles:
        place_img("sprites/hole.png", background, tile)
    return background

def place_stool(background):
    stool_tile = (0, 0)
    place_img("sprites/stool.png", background, stool_tile)
    return background
    
def place_goal(background):
    goal_tile = (3, 3)
    place_img("sprites/goal.png", background, goal_tile)
    return background

def place_agent(background, state_str):
    state, orientation = state_str.split("_")
    state = int(state)
    orientation_str = orientation_map[orientation]
    agent_tile = (int(state % grid_size), int(state / grid_size))
    if agent_tile not in lake_tiles:
        place_img(f"sprites/elf_{orientation_str}.png", background, agent_tile)
    else:
        place_img(f"sprites/cracked_hole.png", background, agent_tile)
    return background

def render_state(state_str, black_ice=False):
    background = generate_background(black_ice=black_ice)
    background = place_lakes(background)
    background = place_stool(background)
    background = place_goal(background)
    background = place_agent(background, state_str)
    return background

if __name__ == '__main__':
    for i in range(16):
        for j in range(4):
            state_str = f"{i}_{j}"
            img = render_state(state_str, black_ice=True)
            img = img.resize((256,256), Image.NEAREST)
            img.save(f"states/dark/images/{state_str}.png")
    
    
