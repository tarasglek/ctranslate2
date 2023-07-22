import sys
from threading import Thread
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import transformers
import os

# num_threads = 4
# torch.set_num_threads(num_threads)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1000"
# device = torch.device("cuda")
USE_GPU=True

nf4_config = BitsAndBytesConfig(load_in_4bit=USE_GPU)

prompt = sys.stdin.read()

name = [
    'WizardLM/WizardCoder-15B-V1.0',
    'emozilla/mpt-7b-storywriter-fast',
    'mosaicml/mpt-7b-instruct',
    'ehartford/WizardLM-30B-Uncensored',
    'ai-forever/ruGPT-3.5-13B',
][0]

tokenizer = AutoTokenizer.from_pretrained(name)
prompt_tokens = tokenizer([prompt], return_tensors="pt")
prompt_token_count = prompt_tokens.input_ids.shape[1]

model = transformers.AutoModelForCausalLM.from_pretrained(
    name,
    # config=config,
    trust_remote_code=True,
    device_map="auto" if USE_GPU else "cpu",
    quantization_config=nf4_config,
)

# model = model.to(device)
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