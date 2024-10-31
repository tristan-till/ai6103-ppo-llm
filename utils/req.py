import requests
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import openai

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
        "prompt": "Where are the character, presents and lakes (only those adjacent to the character) in the scene? Do not describe the way they look. Answer with as little words as possible. E.g. 'Character top left, present bottom right, lake below the character'; 'Character middle-left, present bottom right, lake right of character'",
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


def get_openai_llm(prompt, model="gpt-4", temperature=0.5, max_tokens=150):
    """
    Sends a text prompt to the OpenAI API and returns the response.

    Parameters:
        prompt (str): The text prompt to send to the OpenAI model.
        model (str): The OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo").
        temperature (float): Sampling temperature. Higher values mean more random completions.
        max_tokens (int): The maximum number of tokens to generate in the completion.

    Returns:
        str: The response from the OpenAI model.
    """
    # Retrieve the API key from constants
    api_key = CONST.OPENAI_API_KEY
    if not api_key:
        print("OpenAI API key not found. Please set the OPENAI_API_KEY in constants.py.")
        return ""

    # Configure OpenAI API key
    openai.api_key = api_key

    try:
        print(f"Sending prompt to OpenAI model {model}...")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Please summarize the information in this description. It is a rendered frame of a reinforcement learning algorithm. Focus on the position of the agent, the present. The character must avoid lakes. Answer with as little words as possible (e.g. character bottom left, present bottom right, lake on top of the character)"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        message = response.choices[0].message['content'].strip()
        print("Received response from OpenAI.")
        return message
    except openai.error.OpenAIError as e:
        print(f"OpenAI API request failed: {e}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ""



if __name__ == '__main__':
    prompt = "Is Thomas Mueller a more accomplished footballer than Neymar?"
    # print(get_llama(prompt))
    response = get_openai_llm(prompt)
    if response:
        print(f"\nOpenAI Response:\n{response}")
    else:
        print("Failed to retrieve a response from OpenAI.")