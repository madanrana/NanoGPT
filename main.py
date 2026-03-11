import torch
import config

from model import GPT
from train import train_model
from generate import generate_text


device = "cuda" if torch.cuda.is_available() else "cpu"


text = open("dataset/input.txt").read()

chars = sorted(list(set(text)))

vocab_size = len(chars)


stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)


model = GPT(
    vocab_size,
    config.n_embd,
    config.n_head,
    config.n_layer,
    config.block_size
).to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate
)


train_model(model, data, optimizer, device)


generated = generate_text(
    model,
    start_token=0,
    decode=decode,
    device=device
)


print("\nGenerated Text:\n")
print(generated)