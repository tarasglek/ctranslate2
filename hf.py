import transformers
import sys
from pathlib import Path

"""
limiting threads:
import os
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
"""
name = 'mosaicml/mpt-7b-storywriter'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.max_seq_len = 83968 # (input + output) tokens can now be up to 83968

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  trust_remote_code=True
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

prompt = sys.stdin.read()
# prompt_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
prompt_tokens = tokenizer([prompt], return_tensors="pt")

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(**prompt_tokens, max_length=2000)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))