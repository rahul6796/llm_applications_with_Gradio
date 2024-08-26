

import os
import io
import IPython.display
from PIL import Image
import base64 
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60

import requests, json
from text_generation import Client
from dotenv import load_dotenv
load_dotenv()
hf_api_key = os.environ['HF_API_KEY']



client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Bearer {hf_api_key}"})


prompt = "Has math been invented or discovered?"
client.generate(prompt, max_new_tokens=256).generated_text