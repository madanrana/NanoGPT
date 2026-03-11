import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):

    def __init__(self, n_embd, head_size, block_size):

        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei * C ** -0.5

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        wei = F.softmax(wei, dim=-1)

        v = self.value(x)

        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, n_embd, block_size):

        super().__init__()

        head_size = n_embd // n_head

        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(n_head)]
        )

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)

        return self.proj(out)


class FeedForward(nn.Module):

    def __init__(self, n_embd):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):

        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head, block_size):

        super().__init__()

        self.sa = MultiHeadAttention(n_head, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class GPT(nn.Module):

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):

        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss