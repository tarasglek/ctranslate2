import ctranslate2
import transformers
import sys
import os
import time
"""
extending context length
https://huggingface.co/mosaicml/mpt-7b-storywriter/discussions/29
"""

# export CT2_VERBOSE=1
# os.environ["CT2_VERBOSE"] = "1"

cpu_interthreads = 16
cpu_intrathreads = 4
# generator = ctranslate2.Generator("mosaicml_mpt-7b-storywriter.ct2", device="cpu", inter_threads=cpu_interthreads, intra_threads=cpu_intrathreads)
# generator = ctranslate2.Generator("mpt-7b-instruct.ct2.int8", device="cpu", inter_threads=16, intra_threads=4)
generator = ctranslate2.Generator("emozilla/mpt-7b-storywriter-fast", device="cuda")
# generator = ctranslate2.Generator("mpt-7b-instruct.ct2.int8", device="cuda")

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

print("input text:")
prompt = sys.stdin.read()
prompt_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
#  https://github.com/OpenNMT/CTranslate2/issues/1127#issuecomment-1514401355
print(prompt, end ="", flush=True)
while True:
    start = time.time()
    num_tokens = 0

    results = generator.generate_tokens(
            prompt_tokens,
            # sampling_temperature=0.8,
            sampling_topk=10,
            max_length=700,
        )
    for result in results:
        print(tokenizer.decode(result.token_id), end ="", flush=True)
        num_tokens += 1

    end = time.time()
    print(f"\nTokens per second:{num_tokens / (end - start)} {end-start}s, {num_tokens} tokens", file=sys.stderr)
if False:
    results = generator.generate_batch([prompt_tokens], max_length=700, sampling_topk=10)

    text = tokenizer.decode(results[0].sequences_ids[0])
    print(text)

