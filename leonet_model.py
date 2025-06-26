import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpansionContractionBlock(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4, motor_dim=128):
        super().__init__()
        expanded_dim = hidden_dim * expansion_factor

        self.expand = nn.Linear(hidden_dim, expanded_dim)
        self.motor_proj = nn.Linear(expanded_dim, motor_dim)
        self.cognitive_proj = nn.Linear(expanded_dim, hidden_dim)
        self.contract = nn.Linear(hidden_dim + motor_dim, hidden_dim)

    def forward(self, x):
        x_expanded = self.expand(x)
        motor_out = self.motor_proj(x_expanded)
        cognitive_out = self.cognitive_proj(x_expanded)
        merged = torch.cat([cognitive_out, motor_out], dim=-1)
        x_contracted = self.contract(merged)
        return x_contracted, motor_out


class LeoNetBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, expansion_factor=4, motor_dim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ecb = ExpansionContractionBlock(hidden_dim, expansion_factor, motor_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ecb_out, motor_out = self.ecb(x)
        x = self.norm2(x + ecb_out)
        return x, motor_out


class LeoNet(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, num_layers=6, num_heads=8, motor_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            LeoNetBlock(hidden_dim, num_heads, motor_dim=motor_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.language_head = nn.Linear(hidden_dim, vocab_size)
        self.motor_heads = nn.ModuleList([
            nn.Linear(motor_dim, motor_dim) for _ in range(num_layers)
        ])

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        motor_outputs = []

        for block, motor_head in zip(self.transformer_blocks, self.motor_heads):
            x, motor_out = block(x)
            motor_outputs.append(motor_head(motor_out))

        x = self.final_norm(x)
        logits = self.language_head(x)

        return logits, motor_outputs
