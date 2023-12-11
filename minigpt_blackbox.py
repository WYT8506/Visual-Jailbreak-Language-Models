import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

from PIL import Image

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import requests
from transformers import pipeline
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--image_path", required=True, help="path to the visual input")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
# ========================================
#   Start hateful speech classifier
# ========================================


API_URL = "https://api-inference.huggingface.co/models/Hate-speech-CNERG/dehatebert-mono-english"#"https://api-inference.huggingface.co/models/hubert233/GPTFuzz"
headers = {"Authorization": f"Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "I like you. I love you",
})
print("output: ", output)


pipe = pipeline("text-classification", model="Hate-speech-CNERG/dehatebert-mono-english")
# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')
# ========================================
#             Gradio Setting
# ========================================

# init chat state
def upload_img(img):
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_img(img, chat_state, img_list)
    return chat_state, img_list


def ask(user_message, chat_state):
    chat.ask(user_message, chat_state)
    return chat_state


def answer(chat_state, img_list, num_beams=1, temperature=1.0):
    llm_message  = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]

    return llm_message, chat_state, img_list
def update_image_reinforce(img, sampled_images,hate_values,alpha = 0.01,sigma = 0.001):
    img.requires_grad=True
    grad = 0
    for i in range(len(sampled_images)):
        pi = (1 / (sigma * torch.sqrt(torch.tensor(2 * torch.pi))))**n * torch.exp(-torch.sum((sampled_images[i] - img)**2) / (2 * sigma**2))
        log_pi = torch.log(pi)
        log_pi.backward()
        grad += hate_values[i]*img.grad

    # Compute the derivative with respect to mu
    img+= grad*0.01


import csv
file = open("harmful_corpus/manual_harmful_instructions.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
prompts_eval = []
num = 3#len(data)
for i in range(num):
    prompts_eval.append(data[i][0])

#img = Image.open(args.image_path).convert('RGB')
#img = vis_processor(img).unsqueeze(0).to(model.device)


#chat_state, img_list = upload_img(img)

wandb.login(key = "d27ec80b679b338c64743149bce53d8d428a1ece")
wandb.init(project = "poison_result", name = "blackbox_attack", config = args.__dict__)
wandb.run.log_code(".", include_fn = lambda x: x.endswith("minigpt_blackbox.py"))
print("######## Start Chatting ########")
trials = 3
variance = 0.001
with torch.no_grad():

    img = torch.randn(1, 3, 224, 224).to(model.device)
    best_img = None

    history =[]
    for i in range(100):
        highest_hate = 0
        print("Iteration:",i)
        hate_values = []
        sampled_images = []
        for j in range(trials):
            hate_value = 0
            noise = torch.randn(1, 3, 224, 224).to(model.device)*variance
            new_img = img+noise
            sampled_images.append(new_img)
            for k, prompt_to_eval in enumerate(prompts_eval):

                user_message = prompt_to_eval
                chat_state, img_list = upload_img(new_img)
                print(user_message)
                print('################ Question %d ################' % (k+1))
                chat_state = ask(user_message, chat_state)
                llm_message, chat_state, img_list = answer(chat_state, img_list)
                print('>>> User:', user_message)
                print('\n')

                print('>>> LLM:\n')
                print(llm_message)
                #measure hate 

                output = pipe(llm_message)
                #output = query({
                #"inputs": llm_message,
                #})
                print("output: ", output)
                hate_value += (1-output[0]['score'])/num
            hate_values.append(hate_value)
            #if hate_value >highest_hate:
                #highest_hate = hate_value
                #best_img = new_img
            print("hate_score: ", hate_value)
            print('\n\n')

        print(f"End Iteration:{i}, Avg hate:{hate_values/trials}")
        wandb.log({"epoch/hate": highest_hate})
        img = update_image_reinforce(img, sampled_images,hate_values)
        #history.append(highest_hate)
        #img = best_img
        #print("history: ",history)