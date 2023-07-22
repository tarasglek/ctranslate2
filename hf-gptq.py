from transformers import AutoTokenizer, pipeline, logging, TextIteratorStreamer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
from threading import Thread
import sys

model_name_or_path = "TheBloke/WizardCoder-15B-1.0-GPTQ"
# Or to load it locally, pass the local download path
# model_name_or_path = "/path/to/models/TheBloke_WizardCoder-15B-1.0-GPTQ"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        use_safetensors=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# prompt_template = '''Below is an instruction that describes a task. Write a response that appropriately completes the request

### Instruction: {prompt}

### Response:'''
prompt = sys.stdin.read()
# prompt = prompt_template.format(prompt="How do I sort a list in Ocaml?")

# outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)

# print(outputs[0]['generated_text'])
prompt_tokens = tokenizer([prompt], return_tensors="pt")
prompt_tokens = prompt_tokens.to("cuda")
prompt_token_count = prompt_tokens.input_ids.shape[1]
# prompt_tokens.input_ids.to('cuda')


streamer = TextIteratorStreamer(tokenizer)
generation_kwargs = dict(
    **prompt_tokens, streamer=streamer, max_new_tokens=prompt_token_count + 100
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()
generated_text = ""

for new_text in streamer:
    generated_text += new_text
    print(new_text, end="", flush=True)