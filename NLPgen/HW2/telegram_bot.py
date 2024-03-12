import os
import sys

import fire
import base64
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

import telebot
from utils.prompter import Prompter

from typing import Union
from dataclasses import dataclass, field

import warnings

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


@dataclass
class Params:
    base_model = "nickypro/tinyllama-15M"
    tokenizer_name = "hf-internal-testing/llama-tokenizer"

    weights_path = "weights"

    load_in_8bit = False
    torch_dtype =  torch.float32
    device_map: str = field(init=False)
    low_cpu_mem_usage = True

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    def __post_init__(self):
        if self.ddp:
            self.device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        else:
            self.device_map = "auto"

params = Params()
     

def xor_decrypt_from_str(encrypted_str, key):
    encrypted_bytes = base64.b64decode(encrypted_str.encode('utf-8'))
    key_bytes = key.encode('utf-8')
    full_key_bytes = (key_bytes * (len(encrypted_bytes) // len(key_bytes) + 1))[:len(encrypted_bytes)]
    decrypted_bytes = bytes([encrypted_byte ^ key_byte for encrypted_byte, key_byte in zip(encrypted_bytes, full_key_bytes)])
    decrypted_str = decrypted_bytes.decode('utf-8')

    return decrypted_str

encrypted_telegram_api_key = "BggEAgMHCQUAAgp2cXgEc19oVwdnVgBWZmBbY0ZAYgYCVQFjQXp8XQFcV2AOZQ=="
telegram_api_key = xor_decrypt_from_str(encrypted_telegram_api_key, "007")

context = ""

def main(
    api_key=telegram_api_key
):
    prompter = Prompter()

    tokenizer = LlamaTokenizer.from_pretrained(params.tokenizer_name)

    joey_model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=params.base_model,
        load_in_8bit=params.load_in_8bit,
        torch_dtype=params.torch_dtype,
        device_map=params.device_map,
        low_cpu_mem_usage=params.low_cpu_mem_usage,
    )
    joey_model = PeftModel.from_pretrained(
        joey_model,
        params.weights_path,
        torch_dtype=params.torch_dtype,
        device_map={'': 0},
    )
    joey_model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        joey_model = torch.compile(joey_model)
        if not params.load_in_8bit:
            joey_model.half() 


    def generate_reply(
        text,
        context=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(text, context)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        # Без стриминга инференса
        with torch.no_grad():
            generation_output = joey_model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True).strip()

        return prompter.get_response(output)


    warnings.filterwarnings("ignore")

    bot = telebot.TeleBot(api_key)

    

    @bot.message_handler(commands=["start"])
    def start(m, res=False):
        global context
        contex = "Hi! I'm Joe!"
        bot.send_message(m.chat.id, context)


    @bot.message_handler(content_types=["text"])
    def process_message(message):
        global context
        reply = generate_reply(
            message.text,
            context
        )
        context = reply
        bot.send_message(message.chat.id, reply)

    print("https://t.me/JoeyGeneratorBot")
    bot.polling(none_stop=False)


if __name__ == "__main__":
    fire.Fire(main)


