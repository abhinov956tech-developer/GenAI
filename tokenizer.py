import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

print("Vocab Size", encoder.n_vocab)

text = 'The Sun rise on east'
encoding = encoder.encode(text)
print("Encoded",encoding)

tokens = [976, 11628, 16601, 402, 23557]
decoding = encoder.decode(tokens)
print("Decoded", decoding)