import torch
import sys
import time
torch.set_num_threads(1)
from basaran.model import load_model
# model = 'gpt2'
# model = 'MBZUAI/LaMini-Flan-T5-783M'
# model = 'google/flan-t5-xl'
model = 'mosaicml/mpt-7b-instruct'
# model = 'mosaicml/mpt-7b-storywriter'
load_in_8bit = False
streaming_model = load_model(model,  trust_remote_code=True, load_in_4bit=load_in_8bit,)

print(f"Memory footprint of {model}: {streaming_model.model.get_memory_footprint()}")
print(f"Model input limit: {streaming_model.tokenizer.max_model_input_sizes}")
start = time.time()
num_tokens = 0
for choice in streaming_model(sys.stdin.read(), max_tokens=1000, echo=True):
    print(choice['text'], flush=True, end='')
    num_tokens += 1
    if (choice['finish_reason']):
        print(f"\n finish reason:{choice['finish_reason']}", flush=True)
end = time.time()
print(f"\nTokens per second:{num_tokens / (end - start)} {end-start}s", file=sys.stderr)
# numactl --physcpubind=0-3 

# taskset -c 0-4 python run.py

# sudo numactl --physcpubind=0-3  sudo -u taras python run.py 