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
        "prompt": "This is the environment for a reinforcement agent. The agent, which has green clothing, must reach the present without touching the lakes. Please describe the environment as concise as possible",
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
        "prompt": prompt
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
# Example usage:
# image = "<path_or_image_data>"
# print(get_llava(image))

if __name__ == '__main__':
    prompt = "Is Thomas Mueller a more accomplished footballer than Neymar?"
    print(get_llama(prompt))