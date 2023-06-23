"""
limiting threads:
"""
import sys
from threading import Thread
import torch
use_cpu = False
# if use_cpu:
num_threads = 4
torch.set_num_threads(num_threads)
# else:
devicedevice = torch.device("cuda")
# devicedevice = torch.device("cpu")

from transformers import AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer  # noqa: E402
import transformers  # noqa: E402

device_map = {
        "transformer.word_embeddings": 0,
            "transformer.word_embeddings_layernorm": 0,
                "lm_head": "cpu",
                    "transformer.h": 0,
                        "transformer.ln_f": 0,
                        }

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
prompt = sys.stdin.read()
# prompt_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
prompt_tokens = tokenizer([prompt], return_tensors="pt")
prompt_token_count = prompt_tokens.input_ids.shape[1]
print(f"Input token count: {prompt_token_count}")

name = 'mosaicml/mpt-7b-instruct'
# name = 'ehartford/WizardLM-30B-Uncensored'
# name = 'emozilla/mpt-7b-storywriter-fast'
config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True, load_in_8bit=True)
config.max_seq_len = 83968 # (input + output) tokens can now be up to 83968

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  trust_remote_code=True,
  load_in_8bit=True,
  device_map="auto",
  quantization_config=quantization_config
)
print(f"Memory footprint of {name}: {model.get_memory_footprint()}")
# copied from llama
prompt_tokens = prompt_tokens.to(model.device)


streamer = TextIteratorStreamer(tokenizer)
generation_kwargs = dict(**prompt_tokens, streamer=streamer, max_new_tokens=prompt_token_count + 61)  # noqa: E501
# model.generate(**generation_kwargs)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()
generated_text = ""
for new_text in streamer:
 generated_text += new_text
 print(new_text, end ="", flush=True)

