def stats(start_time, end_time, token_count):
    duration = end_time - start_time
    tokens_per_second = token_count / duration
    return f"{tokens_per_second} tokens/second, {duration}s, {token_count} tokens"
