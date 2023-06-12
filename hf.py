"""
limiting threads:
"""
import torch
use_cpu = False
# if use_cpu:
num_threads = 6
torch.set_num_threads(num_threads)
# else:
device = torch.device("cuda")
import transformers
import sys



from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

prompt = sys.stdin.read()
# prompt_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
prompt_tokens = tokenizer([prompt], return_tensors="pt")
prompt_token_count = prompt_tokens.input_ids.shape[1]
print(f"Input token count: {prompt_token_count}")

name = 'mosaicml/mpt-7b-storywriter'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True, load_in_8bit=True)
config.max_seq_len = 83968 # (input + output) tokens can now be up to 83968

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  trust_remote_code=True,
  load_in_8bit=True,
  device_map="auto",
)
print(f"Memory footprint of {name}: {model.get_memory_footprint()}")


# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(**prompt_tokens, max_length=prompt_token_count+61)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))