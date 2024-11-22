import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import tiktoken
import requests

# Hyperparameters
batch_size = 4 # how many independent sequences will we process in parallel?
context_length = 16 # what is the maximum context length for predictions?
d_model = 64 # embedding dimension
num_heads = 4 # number of attention heads
num_blocks = 8 # number of transformer blocks
learning_rate = 1e-3 # how fast should the model learn?
dropout = 0.1 # dropout rate
max_iters = 50000 # how many iterations to train?
eval_interval = 50 # how often to evaluate the model?
eval_iters = 20 # how many iterations to use while evaluating?
device = 'cuda' if torch.cuda.is_available() else 'cpu' # which device to use?
TORCH_SEED = 1337 # torch seed
torch.manual_seed(TORCH_SEED) # set seed

if not os.path.exists('sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true'
    with open('sales_textbook.txt', 'wb') as f:
        r = requests.get(url).content
        f.write(r)
with open('sales_textbook.txt', 'r') as f:
    text = f.read()

# tokenize
tokenizer = tiktoken.get_encoding('cl100k_base')
tokens = tokenizer.encode(text)

max_token_value = max(tokens)
train_idx = int(0.9 * len(tokens))
train_tokens = torch.tensor(tokens[:train_idx])
valid_tokens = torch.tensor(tokens[train_idx:])

class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model // num_heads)
        self.Wk = nn.Linear(d_model, d_model // num_heads)
        self.Wv = nn.Linear(d_model, d_model // num_heads)

        self.register_buffer('tril', torch.tril(
            torch.ones((context_length, context_length))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_model // num_heads)
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        attn = attn @ V
        return attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([ScaledDotProductAttention() for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention()
        self.ffn = FeedForwardNetwork()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(max_token_value + 1, d_model)
        self.blocks = nn.Sequential(*([TransformerBlock() for _ in range(num_blocks)] + [nn.LayerNorm(d_model)]))
        self.linear = nn.Linear(d_model, max_token_value + 1)

    def forward(self, idx, targets=None):
         B, T = idx.shape # batch size, sequence length
         position_encoding_lookup_table = torch.zeros(context_length, d_model, device=device)
         position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
         position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
         position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)

         position_embedding = position_encoding_lookup_table[:T, :].to(device)

         x = self.token_embedding_table(idx) + position_embedding
         x = self.blocks(x)

         logits = self.linear(x)

         if targets is not None:
             B, T, C = logits.shape
             loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
         else:
             loss = None
         return logits, loss
    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -context_length:]
            logits, loss = self(idx_crop)
            logits_last = logits[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = Model().to(device)

def get_batch(split):
    data = train_tokens if split == 'train' else valid_tokens
    i = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in i]).to(device)
    y = torch.stack([data[i+1:i+context_length+1] for i in i]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
tracked_loss = list()

for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_loss.append(losses)
        print(f'step {step}: train loss {losses["train"].item():.4f}, val loss {losses["val"].item():.4f}')

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save the model
torch.save(model.state_dict(), 'sales_textbook_model.pt')

# evaluate model
model.eval()
start = "The salesperson"
start_idx = tokenizer.encode(start)
x = (torch.tensor(start_idx, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print(tokenizer.decode(y[0].tolist()))