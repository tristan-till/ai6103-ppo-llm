import requests
import utils.constants as CONST

def get_llava(payload):
    res = requests.post(f"{CONST.LLAVA_BASE_URL}/path", data=payload)
    return res

def get_llama(payload):
    res = requests.post(f"{CONST.LLAMA_BASE_URL}/path", data=payload)
    return res