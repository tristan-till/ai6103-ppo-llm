import requests
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.constants as CONST

def get_llava(image):
    """
    Sends an image prompt to the Ollama API (LLaVA model) and returns the response.

    Parameters:
        image (str): The image data or path for LLaVA to analyze.

    Returns:
        str: The response from the LLaVA model.
    """
    payload = {
        "model": "llava:7b",
        "prompt": "You will be presented with a 2d-scene of a agent, a present and lakes. The agent must walk towards the present and can move up, down, right and left. It is not allowed to fall into the lake. Please answer short and concisely. Where is the agent, where is the present, where are the lakes in the scene? E.g. 'Agent top left, present bottom right, lakes in the top right, center and bottom left'; 'Agent bottom left, present bottom right, two lakes bottom left'; 'Agent top right, present top left, three lakes in the center'",
        "images": [image]
    }

    response = requests.post(CONST.OLLAMA_BASE_URL, json=payload)

    full_response = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            full_response += data.get('response', '')
            if data.get('done', False):
                break

    return full_response.strip()

def get_llama(prompt):
    """
    Sends a text prompt to the Ollama API (LLaMA model) and returns the response.

    Parameters:
        prompt (str): The text prompt to send to LLaMA.

    Returns:
        str: The response from the LLaMA model.
    """
    payload = {
        "model": "llama3.2:1b",
        "prompt": f"Please summarize the information in this description. It is a rendered frame of a reinforcement learning algorithm. Focus on the position of the agent, the lakes and the present. The character must avoid lakes. Answer with as little words as possible (e.g. 'Agent bottom left, present bottom right, lake on top of the agent'; 'Agent top right, present bottom left, no lake next to agent'; 'Agent top left, present top left, lake above and below agent'): {prompt}"
    }

    response = requests.post(CONST.OLLAMA_BASE_URL, json=payload, stream=True)
    
    full_response = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            full_response += data.get('response', '')
            if data.get('done', False):
                break
    
    return full_response.strip()

if __name__ == '__main__':
    prompt = "Is Thomas Mueller a more accomplished footballer than Neymar?"
    print(get_llama(prompt))