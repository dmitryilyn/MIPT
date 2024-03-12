import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
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
     

def main(
    server_name: str = "127.0.0.1",
    share_gradio: bool = True,
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


    def generate_reply_async(
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

        # Асинхронный вывод
        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault("stopping_criteria", transformers.StoppingCriteriaList())
            kwargs["stopping_criteria"].append(Stream(callback_func=callback))
            with torch.no_grad():
                joey_model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(generate_with_callback, kwargs, callback=None)

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                decoded_output = tokenizer.decode(output)

                if output[-1] in [tokenizer.eos_token_id]:
                    break

                yield prompter.get_response(decoded_output)

        return


    warnings.filterwarnings("ignore")

    iface = gr.Interface(
       fn=generate_reply_async,
       inputs=[
            gr.components.Textbox(
                lines=1,
                label="Ваше сообщение",
                placeholder="Hi!",
            ),
            gr.components.Textbox(
                lines=1,
                label="Контекст"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Температура"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=1024, step=1, value=512, label="Максимальное число токенов"
            )
        ],
        outputs=[
            gr.components.Textbox(
                lines=5,
                label="Ответ",
            )
        ]
    )
    iface.queue()
    iface.launch(server_name=server_name, share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)


