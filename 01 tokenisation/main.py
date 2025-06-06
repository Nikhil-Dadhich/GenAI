import tiktoken

enc = tiktoken.encoding_for_model('gpt-4o')

text = "Hello, world! This is a test of the tiktoken library."

tokens = enc.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")

decoded_text = enc.decode(tokens)
print(f"Decoded Text: {decoded_text}")
print(f"Number of tokens: {len(tokens)}")