import torch
from model import Model

model = Model()
model.load_state_dict(torch.load("sales_textbook_model.pt"))

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"param num: {total_params}")

