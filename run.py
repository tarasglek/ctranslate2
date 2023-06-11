import ctranslate2
import transformers

generator = ctranslate2.Generator("limcheekin/mpt-7b-storywriter-ct2")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

prompt = "Long long time ago, "
tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
# 
results = generator.generate_batch([tokens], max_length=256, sampling_topk=10)
# 
text = tokenizer.decode(results[0].sequences_ids[0])
