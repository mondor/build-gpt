import tiktoken

SPECIAL_TOKENS = {
    '<|bos|>': 50256,  # same as <|endoftext|>
    '<|user_start|>': 50257,
    '<|user_end|>': 50258,
    '<|assistant_start|>': 50259,
    '<|assistent_end|>': 50260,
}

def render_conversation():
    pass