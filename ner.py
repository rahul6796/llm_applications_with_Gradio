

import os
import io
import base64
from dotenv import load_dotenv
import json
import requests
from transformers import pipeline
import gradio as gr


load_dotenv()

hf_api_key = os.getenv('HF_API_KEY')


def get_completion(inputs, parameters=None, ENDPOINT_URL = ""):
    try:
        headers = {
            'Authorization': f'Bearer {hf_api_key}',
            'Content-Type': 'application/json'
        }

        data  = {'inputs': inputs}

        if parameters is not None:
            data.update({'parameters': parameters})

        response = requests.request('POST',
                                    ENDPOINT_URL,
                                    headers=headers,
                                    data=json.dumps(data))
        
        return json.loads(response.content.decode('utf-8'))
    
    except Exception as e:
        print(f'error is raised from get complete method :: {e}')


get_completion = pipeline('ner', model='dslim/distilbert-NER', aggregation_strategy='average')


# def merge_tokens(tokens):
#     merged_tokens = []
#     for token in tokens:
#         if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
#             last_token = merged_tokens[-1]
#             last_token['word'] += token['word'].replace('##', '')
#             last_token['end'] = token['end']
#             last_token['score'] = (last_token['score'] + token['score']) / 2
#         else:
#             merged_tokens.append(token)

#     return merged_tokens

def ner(input):
    output = get_completion(input)
    # merged_token = merge_tokens(output)

    return {'text': input, 
            'entities': output}



demo = gr.Interface(
    fn = ner, 
    inputs = [gr.Textbox(label = "Text to find entities!", lines=2)],
    outputs = [gr.HighlightedText(label='Entities')],
    title = "Named Entity Recognition",
    description='Using DistilBert model for NER.',
    allow_flagging='never'
)

demo.launch()


