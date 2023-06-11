```
pip install -qU ctranslate2 transformers[torch] accelerate einops
git clone https://huggingface.co/mosaicml/mpt-7b-instruct
time ct2-transformers-converter --model mpt-7b-instruct/ --output_dir "mpt-7b-instruct.ct2.int8" --trust_remote_code --low_cpu_mem_usage --quantization int8
time python run.py < input.txt
```