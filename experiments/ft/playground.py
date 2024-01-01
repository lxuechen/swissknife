"""
Single-turn chatbot playground to test our models.
"""
import abc
import dataclasses
import logging
from threading import Thread

import fire
import gradio as gr
import torch
import transformers

TEXT_FORMATTER = {'function_calling', 'guanaco_oasst', 'alpaca'}


class TextFormatter(abc.ABC):
    @abc.abstractmethod
    def __call__(self, dict_data: dict):
        raise NotImplementedError


@dataclasses.dataclass
class FunctionCallingTextFormatter(TextFormatter):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, dict_data: dict):
        system, chat = dict_data['system'], dict_data['chat']
        text = f"{system}\n\n{chat}"
        text = text.replace('<|endoftext|>', self.tokenizer.eos_token)
        return text


def main(
    model_name_or_path: str = "microsoft/phi-2",
    cache_dir=None,
    max_new_tokens=1024,
    num_beams=1,
    tf32=True,
    show_api=True,
    share=False,
    debug=False,
    server_name="0.0.0.0"
):
    # python playground.py --model_name_or_path "/self/scr-ssd/lxuechen/working_dir/phi-2-tool-use"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = tf32

    # On MacOS, loads on mps; on GPU-enabled Linux, loads on GPU 0.
    model: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True,
        trust_remote_code=True
    )
    streamer = transformers.TextIteratorStreamer(tokenizer)

    @torch.inference_mode()
    def predict(message, history, system, temperature, top_p):
        # TODO: Fix message formatting so history is not omitted.
        text_formatter = FunctionCallingTextFormatter(tokenizer=tokenizer)
        prompt = text_formatter(dict_data={'system': system, 'chat': message})
        logging.warning(f"Formatted prompt:\n{prompt}")

        generation_kwargs = dict(
            inputs=tokenizer(prompt, return_tensors="pt").input_ids.to(model.device),
            generation_config=transformers.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0.0,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ),
            streamer=streamer
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        response = ""
        for token in streamer:
            response += token
            yield response

    with gr.Blocks() as demo:
        with gr.Row():
            temperature_box = gr.Slider(0.0, 1.0, 0.7, label="temperature")
            top_p_box = gr.Slider(0.0, 1.0, 0.9, label="top p")

        with gr.Row():
            system_box = gr.Textbox(show_label=False, placeholder="Enter your system message.")

        gr.ChatInterface(predict, additional_inputs=[system_box, temperature_box, top_p_box])

    demo.launch(share=share, debug=debug, show_api=show_api, server_name=server_name)


if __name__ == "__main__":
    fire.Fire(main)
