import sys
from threading import Thread
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import transformers
import os

# num_threads = 4
# torch.set_num_threads(num_threads)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1000"
# device = torch.device("cpu")
USE_GPU=False

nf4_config = BitsAndBytesConfig(load_in_4bit=USE_GPU)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

prompt = sys.stdin.read()
prompt_tokens = tokenizer([prompt], return_tensors="pt")
prompt_token_count = prompt_tokens.input_ids.shape[1]

name = [
    'emozilla/mpt-7b-storywriter-fast',
    'mosaicml/mpt-7b-instruct',
    'ehartford/WizardLM-30B-Uncensored'
][0]
config = transformers.AutoConfig.from_pretrained(
    name, trust_remote_code=True, quantization_config=nf4_config
)
config.max_seq_len = 83968

model = transformers.AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    trust_remote_code=True,
    device_map="auto" if USE_GPU else "cpu",
    quantization_config=nf4_config,

)
# model = model.to("cpu")
prompt_tokens = prompt_tokens.to(model.device)
streamer = TextIteratorStreamer(tokenizer)
generation_kwargs = dict(
    **prompt_tokens, streamer=streamer, max_new_tokens=prompt_token_count + 61
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()
generated_text = ""

for new_text in streamer:
    generated_text += new_text
    print(new_text, end="", flush=True)