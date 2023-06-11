import ctranslate2
import transformers
import sys
generator = ctranslate2.Generator("mosaicml_mpt-7b-storywriter.ct2", device="cpu", inter_threads=4, intra_threads=4)

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

print("input text:")
prompt = sys.stdin.read()
tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
# 
results = generator.generate_batch([tokens], max_length=256, sampling_topk=10)
# 
text = tokenizer.decode(results[0].sequences_ids[0])
print(text)
