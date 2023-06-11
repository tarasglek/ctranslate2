pip install -qU ctranslate2 transformers[torch] accelerate einops
time ct2-transformers-converter --model mosaicml_mpt-7b-storywriter/ --output_dir mosaicml_mpt-7b-storywriter.ct2 --trust_remote_code --low_cpu_mem_usage --quantization float16
