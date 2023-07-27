import time
load_time = time.time()
import measure
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
    "cerebras/btlm-3b-8k-base",
    'mosaicml/mpt-7b-instruct',
    'WizardLM/WizardCoder-15B-V1.0',
    'emozilla/mpt-7b-storywriter-fast',
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

streamer = TextIteratorStreamer(tokenizer)
generation_kwargs = dict(
    **prompt_tokens, streamer=streamer, max_new_tokens=8000 - prompt_token_count,
    # num_beams=1,
    # max_new_tokens=50,
    # early_stopping=True,
    # no_repeat_ngram_size=2
)

start_time = time.time()
first_response_time = None
response_token_count = 0

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()
generated_text = ""

for new_text in streamer:
    if first_response_time is None:
        first_response_time = time.time()
    response_token_count = response_token_count + 1
    generated_text += new_text
    print(new_text, end="", flush=True)

end_time = time.time()

print(f"\nPrompt: {measure.stats(start_time, first_response_time, prompt_token_count)}", file=sys.stderr)
print(f"Response: {measure.stats(first_response_time, end_time, response_token_count)}", file=sys.stderr)
print(f"Overall: {measure.stats(load_time, end_time, prompt_token_count+response_token_count)}", file=sys.stderr)
print(f"Overall-ResponseTokens: {measure.stats(load_time, end_time, response_token_count)}", file=sys.stderr)
