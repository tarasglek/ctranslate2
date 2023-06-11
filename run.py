import ctranslate2
import transformers
import sys
import os
import time
# export CT2_VERBOSE=1
os.environ["CT2_VERBOSE"] = "1"

generator = ctranslate2.Generator("mosaicml_mpt-7b-storywriter.ct2", device="cpu", inter_threads=4, intra_threads=4)
# generator = ctranslate2.Generator("mpt-7b-storywriter-ct2.fp8", device="cuda")

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

print("input text:")
prompt = sys.stdin.read()
prompt_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
#  https://github.com/OpenNMT/CTranslate2/issues/1127#issuecomment-1514401355
if False:
    start = time.time()
    num_tokens = 0

    results = generator.generate_tokens(
            prompt_tokens,
            # sampling_temperature=0.8,
            sampling_topk=10,
            max_length=2048,
        )
    for result in results:
        print(result.token, end =" ", flush=True)
        num_tokens += 1

    end = time.time()
    print("Tokens per second:", num_tokens / (end - start))
else:
    results = generator.generate_batch([prompt_tokens], max_length=700, sampling_topk=10)

    text = tokenizer.decode(results[0].sequences_ids[0])
    print(text)

