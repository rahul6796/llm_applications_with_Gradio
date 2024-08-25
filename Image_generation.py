import os
import io
import IPython.display
from PIL import Image
import base64 
from dotenv import load_dotenv
load_dotenv()


hf_api_key = os.getenv('HF_API_KEY')
hf_api_img = os.getenv('HF_API_ITT_IMG')

import requests, json



def get_completion(inputs, parameters=None, ENDPOINT_URL=hf_api_img):
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }   
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL,
                                headers=headers,
                                data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))


# prompt = "a dog in a park"

# result = get_completion(prompt)
# IPython.display.HTML(f'<img src="data:image/png;base64,{result}" />')

import gradio as gr 

def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

def generate(prompt):
    output = get_completion(prompt)
    result_image = base64_to_pil(output)
    return result_image

gr.close_all()
demo = gr.Interface(fn=generate,
                    inputs=[gr.Textbox(label="Your prompt")],
                    outputs=[gr.Image(label="Result")],
                    title="Image Generation with Stable Diffusion",
                    description="Generate any image with Stable Diffusion",
                    allow_flagging="never",
                    examples=["the spirit of a tamagotchi wandering in the city of Vienna","a mecha robot in a favela"])

demo.launch()