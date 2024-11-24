
# api/ollama_api.py

import requests
import json
from config import OLLAMA_BASE_URL

def get_llava(image_base64):
    """
    Sends an image prompt to the Ollama API (LLaVA model) and returns the response.
    """
    payload = {
        "model": "llava:7b",
        "prompt": "Where are the character, presents, and lakes (only those adjacent to the character) in the scene? Do not describe the way they look. Answer with as few words as possible.",
        "images": [image_base64]
    }
    try:
        response = requests.post(OLLAMA_BASE_URL, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                full_response += data.get('response', '')
                if data.get('done', False):
                    break
        return full_response.strip()
    except Exception as e:
        print(f"Error in get_llava: {e}")
        return ""

def get_llama(prompt):
    """
    Sends a text prompt to the Ollama API (LLaMA model) and returns the response.
    """
    payload = {
        "model": "llama3.2:1b",
        "prompt": f"Please summarize the information in this description. It is a rendered frame of a reinforcement learning algorithm. Focus on the position of the agent and the presents. The character must avoid lakes. Answer with as few words as possible: {prompt}"
    }
    try:
        response = requests.post(OLLAMA_BASE_URL, json=payload, stream=True)
        response.raise_for_status()
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                full_response += data.get('response', '')
                if data.get('done', False):
                    break
        return full_response.strip()
    except Exception as e:
        print(f"Error in get_llama: {e}")
        return ""
