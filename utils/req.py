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
        "prompt": """Where are the character, presents and lakes (only those adjacent to the character) in the scene? 
        Do not describe the way they look. Answer with as little words as possible. 
        E.g. 'Character top left, present bottom right, lake below the character'; 
        'Character middle-left, present bottom right, lake right of character'""",
        "images": [image],
        "options": {
            "seed": 42,
            "top_k": 20,
            "top_p": 0.9,
            "min_p": 0.0,
            "temperature": 0.8,
  }
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
        "prompt": f"Please summarize the information in this description. It is a rendered frame of a reinforcement learning algorithm. Focus on the position of the agent, the present. The character must avoid lakes. Answer with as little words as possible (e.g. character bottom left, present bottom right, lake on top of the character): {prompt}"
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