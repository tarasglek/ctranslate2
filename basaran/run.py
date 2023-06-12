import torch
import sys
torch.set_num_threads(1)
from basaran.model import load_model
# model = 'gpt2'
model = 'google/flan-t5-xl'
# model = 'mosaicml/mpt-7b-instruct'
# model = 'mosaicml/mpt-7b-storywriter'
load_in_8bit = True
streaming_model = load_model(model,  trust_remote_code=True, load_in_8bit=load_in_8bit,)

print(f"Memory footprint of {model}: {streaming_model.model.get_memory_footprint()}")


for choice in streaming_model(sys.stdin.read(), max_tokens=1000, echo=True):
    print(choice['text'], flush=True, end='')
    if (choice['finish_reason']):
        print(f"\n finish reason:{choice['finish_reason']}", flush=True)

# numactl --physcpubind=0-3 

# taskset -c 0-4 python run.py

# sudo numactl --physcpubind=0-3  sudo -u taras python run.py 