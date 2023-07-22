from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse

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

prompt_template = '''Below is an instruction that describes a task. Write a response that appropriately completes the request

### Instruction: {prompt}

### Response:'''
prompt = prompt_template.format(prompt="How do I sort a list in Ocaml?")

outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)

print(outputs[0]['generated_text'])